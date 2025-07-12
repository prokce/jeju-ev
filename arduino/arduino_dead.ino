// ──────────────────────────────────────────────────────────────────────────────
//  Arduino GIGA
//  ▸ MPU9250 DMP6 YPR 퍼블리셔       (geometry_msgs/Vector3, topic: imu/data)
//  ▸ Steering 서보모터 서브스크라이버 (std_msgs/Float32,   topic: steering_angle)
//  ▸ 구동모터 속도  서브스크라이버   (std_msgs/Int32,     topic: motor_speed)
//      – BTN8982TA 듀얼 하프브리지 1개 → 풀 H-브리지 구동 (FWD 전진 전용)
//      – 0-255  PWM 값으로 속도 제어
//  2025-07-03  (spin_some 추가 버전)
// ──────────────────────────────────────────────────────────────────────────────
#include <Servo.h>

#include "I2Cdev.h"
#include "MPU9250_9Axis_MotionApps41.h"
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
  #include <Wire.h>
#endif

#include <micro_ros_arduino.h>
#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h> 
#include <geometry_msgs/msg/vector3.h>
#include <std_msgs/msg/float32.h>
#include <std_msgs/msg/int32.h>
#include <std_msgs/msg/int64.h>

#include <math.h>

#include <std_msgs/msg/string.h>            // ★ 추가
#include <rosidl_runtime_c/string_functions.h>  // ★ 추가

// ─────────────── PIN 매핑 ───────────────
const uint8_t SERVO_PIN = 2;   // 스티어링 서보 (PWM2)

const uint8_t IN1_PIN  = 8;    // BTN8982 IN_1   (PWM8)  – 방향
const uint8_t IN2_PIN  = 9;    // BTN8982 IN_2   (PWM9)  – 방향
const uint8_t INH1_PIN = 3;    // BTN8982 INH_1  (PWM3)  – 속도 PWM
const uint8_t INH2_PIN = 4;    // BTN8982 INH_2  (PWM4)  – 속도 PWM

const uint8_t MPU_INT_PIN = 6; // MPU9250 INT
const uint8_t LED_PIN     = 13;

// ─────────────── 서보 매핑 ───────────────
// 차량 기준 휠 조향각 −20°‥+20° ↔ 서보 명령 0‥40
const int   SERVO_RIGHT_CMD  = 40;
const int   SERVO_CENTER_CMD = 20;
const int   SERVO_LEFT_CMD   = 0;
const float MAX_WHEEL_DEG    = 20.0f;

Servo steering;

// ────────── Encoder PIN ────────── 
//E30S4-1024-3-T-5    
const uint8_t ENC_PIN_A = 42;   // A상
const uint8_t ENC_PIN_B = 43;   // B상
const uint8_t ENC_PIN_Z = 44;   // Z상
// ───────────────────────────────────────  

// ─────────────── MPU9250 객체/버퍼 ───────────────
MPU9250   mpu;
bool      dmpReady   = false;
volatile bool mpuInterrupt = false;

volatile bool imu_ready = false;
rcl_timer_t   imu_timer;

#define IMU_PERIOD_MS 10   // 100 Hz

uint8_t   mpuIntStatus;
uint8_t   devStatus;
uint16_t  packetSize;
uint16_t  fifoCount;
uint8_t   fifoBuffer[64];
Quaternion  q;
VectorFloat gravity;
float       ypr[3];
float yaw_filtered = 0.0f;    // 필터링된 yaw
const float alpha = 0.05f;    // 로우패스 필터 계수 (작을수록 부드러움)

//______엔코더 전역_________________________________
volatile int64_t encoderCount   = 0;
int64_t           lastEncoderCount = 0;   // (디버그용)
rcl_publisher_t encoder_pub;           // ROS 퍼블리셔
std_msgs__msg__Int64 encoder_msg;
rcl_timer_t     encoder_timer;         // 100 ms 타이머

//__________total________________________
// 최신 상태 저장
int   g_pwm   = 0;
int   g_speed = 0;
float g_angle = 0.0f;
float g_yaw   = 0.0f;

// /total 퍼블리셔
rcl_publisher_t total_pub;
std_msgs__msg__String total_msg;
rcl_timer_t     total_timer;         // 100 ms (10 Hz)



// ─────────────── MPU ISR ───────────────
#if defined(ESP32) || defined(ESP8266)
  #define ISR_ATTR IRAM_ATTR
#else
  #define ISR_ATTR
#endif
void ISR_ATTR dmpDataReady() {mpuInterrupt = true;}

// Encoder ISR 2종
void ISR_ATTR updateEncoder()
{
  int a = digitalRead(ENC_PIN_A);
  int b = digitalRead(ENC_PIN_B);
  encoderCount += (a == b) ? 1 : -1;
}
// void ISR_ATTR resetEncoder()
// {
//   encoderCount = 0;
// }

// ─────────────── micro-ROS 객체 ───────────────
rcl_allocator_t         allocator;
rclc_support_t          support;
rcl_node_t              node;
rcl_publisher_t         ypr_pub;
rcl_subscription_t      angle_sub;
rcl_subscription_t      speed_sub;
rclc_executor_t         executor;

geometry_msgs__msg__Vector3 ypr_msg;
std_msgs__msg__Float32       angle_msg;
std_msgs__msg__Int32         speed_msg;


// ───────────────── 유틸 함수 ─────────────────
void set_steering_angle(float wheel_deg)
{
  wheel_deg = fmaxf(-MAX_WHEEL_DEG, fminf(MAX_WHEEL_DEG, wheel_deg));
  int cmd = (int)roundf(-wheel_deg + SERVO_CENTER_CMD);
  cmd = constrain(cmd, SERVO_LEFT_CMD, SERVO_RIGHT_CMD);
  steering.write(cmd);
}

void set_motor_speed(int pwm)
{
  pwm = constrain(pwm, 0, 255);
  digitalWrite(INH1_PIN, HIGH);      // 전진 방향
  digitalWrite(INH2_PIN, HIGH); 
  digitalWrite(IN1_PIN, HIGH);
  digitalWrite(IN2_PIN, LOW); 
          
  analogWrite(IN1_PIN, pwm);       // 속도 PWM은 여기
  g_pwm = pwm;
}

// ───────────────── 콜백 ─────────────────
void angle_callback(const void * msg_in)
{
  auto msg = static_cast<const std_msgs__msg__Float32 *>(msg_in);
  g_angle = msg->data;
  set_steering_angle(msg->data);
}

void speed_callback(const void * msg_in)
{
  auto msg = static_cast<const std_msgs__msg__Int32 *>(msg_in);
  g_speed = msg->data;
  set_motor_speed(msg->data);
}

void imu_timer_cb(rcl_timer_t*, int64_t)
{
  read_and_publish_imu();
}

// Encoder 주기 콜백
void encoder_timer_cb(rcl_timer_t*, int64_t)
{
  encoder_msg.data = static_cast<int64_t>(encoderCount);
  rcl_publish(&encoder_pub, &encoder_msg, nullptr);
}

// ───────── IMU 처리 함수 (100 Hz 타이머 콜백용) ─────────
void read_and_publish_imu()
{
  if (!dmpReady) return;                       // DMP 미준비 시 무시

  // 새 패킷이 없으면 빠르게 리턴
  uint16_t fifoCount = mpu.getFIFOCount();
  if (!imu_ready && fifoCount < packetSize) return;
  imu_ready = false;                           // 플래그 클리어

  /* --- FIFO 상태 확인 --------------------------------------------------- */
  mpuIntStatus = mpu.getIntStatus();
  fifoCount    = mpu.getFIFOCount();

  // 오버플로우 처리
  if ((mpuIntStatus & 0x10) || fifoCount == 1024) {
    mpu.resetFIFO();
    return;
  }
  // DMP 데이터레디 비트 아니면 무시
  if (!(mpuIntStatus & 0x02)) return;

  /* --- 패킷 읽기 --------------------------------------------------------- */
  while (fifoCount < packetSize) fifoCount = mpu.getFIFOCount();
  mpu.getFIFOBytes(fifoBuffer, packetSize);
  fifoCount -= packetSize;

  /* --- Yaw Pitch Roll 계산 ---------------------------------------------- */
  mpu.dmpGetQuaternion(&q, fifoBuffer);
  mpu.dmpGetGravity(&gravity, &q);
  mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);

  float yaw   = ypr[0] * 180.0f / M_PI;
  float pitch = ypr[1] * 180.0f / M_PI;
  float roll  = ypr[2] * 180.0f / M_PI;

  // === 로우패스 필터 적용 ===
  //yaw_filtered = alpha * yaw + (1.0f - alpha) * yaw_filtered;

  // === ROS 퍼블리시 ===
  //ypr_msg.x = yaw_filtered;
  ypr_msg.x = yaw;
  ypr_msg.y = pitch;
  ypr_msg.z = roll;
  rcl_publish(&ypr_pub, &ypr_msg, nullptr);

  static bool led = false;
  digitalWrite(LED_PIN, led = !led);

  g_yaw = yaw;
  //g_yaw = yaw_filtered;
}

void total_timer_cb(rcl_timer_t*, int64_t)
{
  /* 버퍼에 YAML-like 문자열 작성 */
  char buf[128];
  int len = snprintf(buf, sizeof(buf),
      "pwm: %d\nangle: %.2f\nencoder: %lld\nspeed: %d\nyaw: %.2f",
      g_pwm, g_angle, encoderCount, g_speed, g_yaw);

  /* micro-ROS String 메시지 채우기 */
  if (total_msg.data.data) {   // 이전 메모리 해제
    std_msgs__msg__String__fini(&total_msg);
  }
  std_msgs__msg__String__init(&total_msg);
  rosidl_runtime_c__String__assignn(&total_msg.data, buf, len);

  rcl_publish(&total_pub, &total_msg, nullptr);
}



// ─────────────── micro-ROS 초기화 ───────────────
void init_microros()
{
  set_microros_transports();     // USB-CDC
  delay(1500);                   // 에이전트 연결 대기

  allocator = rcl_get_default_allocator();
  rclc_support_init(&support, 0, NULL, &allocator);

  rclc_node_init_default(&node, "mpu9250_node", "", &support);

  rclc_publisher_init_default(
      &ypr_pub,
      &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Vector3),
      "imu/data");

  // 엔코더 퍼블리셔
  rclc_publisher_init_default(
      &encoder_pub, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int64),
      "encoder/count");

  rclc_publisher_init_default(
    &total_pub, &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String),
    "/status/total");

  rclc_subscription_init_default(
      &angle_sub,
      &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
      "steering_angle");

  rclc_subscription_init_default(
      &speed_sub,
      &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32),
      "motor_speed");

  /* ⑤ executor: 핸들 수*/
  rclc_executor_init(&executor, &support.context, 5, &allocator);

  //angle 구독
  rclc_executor_add_subscription(
      &executor, &angle_sub, &angle_msg, &angle_callback, ON_NEW_DATA);

  //pwm 구독
  rclc_executor_add_subscription(
      &executor, &speed_sub, &speed_msg, &speed_callback, ON_NEW_DATA);

    /* ⑦ IMU 타이머(100 Hz) 생성 & 등록 */
  rclc_timer_init_default(
      &imu_timer, &support,
      RCL_MS_TO_NS(IMU_PERIOD_MS),        // 10 ms
      imu_timer_cb);
  rclc_executor_add_timer(&executor, &imu_timer);


  /* ─── 엔코더 타이머 ─── */
  rclc_timer_init_default(
      &encoder_timer, &support,
      RCL_MS_TO_NS(100),          // 100 ms = 10 Hz
      encoder_timer_cb);
  rclc_executor_add_timer(&executor, &encoder_timer);

  /* ② total 타이머(10 Hz) */
  rclc_timer_init_default(
      &total_timer, &support,
      RCL_MS_TO_NS(100),               // 100 ms
      total_timer_cb);
  rclc_executor_add_timer(&executor, &total_timer);
}



// ─────────────── Arduino setup ───────────────
void setup()
{
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
  Wire.begin();
  Wire.setClock(400000);
#endif

  Serial.begin(115200);
  while (!Serial) {}

  pinMode(IN1_PIN, OUTPUT);
  pinMode(IN2_PIN, OUTPUT);
  pinMode(INH1_PIN, OUTPUT);
  pinMode(INH2_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);

  /* ───── Encoder pinMode & 인터럽트 ───── */   
  pinMode(ENC_PIN_A, INPUT_PULLUP);
  pinMode(ENC_PIN_B, INPUT_PULLUP);
  pinMode(ENC_PIN_Z, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENC_PIN_A), updateEncoder, CHANGE);
  //attachInterrupt(digitalPinToInterrupt(ENC_PIN_Z), resetEncoder , RISING);
/* ─────────────────────────────────────── */ 

  steering.attach(SERVO_PIN);
  set_steering_angle(0.0f);

  init_microros();

  mpu.initialize();
  pinMode(MPU_INT_PIN, INPUT);

  devStatus = mpu.dmpInitialize();

  if (devStatus == 0) {
    mpu.setDMPEnabled(true);
    attachInterrupt(digitalPinToInterrupt(MPU_INT_PIN),
                    dmpDataReady, RISING);
    packetSize = mpu.dmpGetFIFOPacketSize();
    dmpReady   = true;
  } else {
    while (1) { digitalWrite(LED_PIN, !digitalRead(LED_PIN)); delay(150); }
  }

  set_motor_speed(0);  // 모터 정지
}

// ─────────────── Arduino loop ───────────────
void loop()
{
  /* executor 가 타이머·구독 이벤트를 모두 처리 */
  rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10));
}