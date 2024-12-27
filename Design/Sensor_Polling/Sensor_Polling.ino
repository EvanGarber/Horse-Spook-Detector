#include "FastIMU.h"
#include <Wire.h>
#include <Adafruit_MLX90614.h>
#include <esp_now.h>
#include <SPI.h>
#include <SD.h>
#include <WiFi.h>

#define IMU_ADDRESS_1 0x68
#define IMU_ADDRESS_2 0x69

#define ONBOARD_LED 2

#define ADDR_PIN_IMU_L 32
#define ADDR_PIN_IMU_R 16

// Setup:
// SCL: 22
// SDA: 21

// ADDR IMU L: 32
// ADDR IMU R: 16

// MOSI: 23
// MISO: 19
// SCK: 18
// CS: 5

// USE 3.3 V FOR EVERYTHING APART FROM SD CARD READER

// -------- ESP_NOW config --------

typedef struct imu_record {
  float ax;
  float ay;
  float az;
  float gx;
  float gy;
  float gz;
  float temp;
} imu_record;

typedef struct struct_message {
  unsigned long mils;
  imu_record imus[3];
  float t_obj;
  float t_amb;
} struct_message;

// Datapacket to send over ESP Now
struct_message dataPacket;

uint8_t broadcastAddress[] = {0x0C, 0x8B, 0x95, 0xA6, 0xFE, 0xB8};


// -------- Sensor config --------

const int chipSelect = 5;
// IMU 1 is for the center IMU
MPU6500 IMU_1;
// IMU 2 is for L and R IMUs, round-robin connected via GPIO pin activation of ADD
// This gets around the issue of the limited I2C addresses on the MPU-6500
MPU6500 IMU_2;
Adafruit_MLX90614 mlx = Adafruit_MLX90614();

calData calib = { 0 };  // Calibration data
AccelData accelData;    // Sensor data
GyroData gyroData;
float IMU_temperature;

File dataFile;


void setup() {
  Wire.begin();
  Wire.setClock(100000); //100khz I2C clock - MLX90614 runs on MSBus which has a lower max speed than I2C - Cannot run at 400khz (haven't tested upper limit)
  Serial.begin(115200);
  while (!Serial) {
    ;
  }

  // Init esp now
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    while (1);
  }
  // Register peer
  esp_now_peer_info_t peerInfo;
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;
  // Add peer        
  if (esp_now_add_peer(&peerInfo) != ESP_OK){
    Serial.println("Failed to add peer");
    while (1);
  }

  // Init IMU 1
  int err1 = IMU_1.init(calib, IMU_ADDRESS_1);

  if (err1 != 0) {
    Serial.print("Error initializing IMU 1: ");
    Serial.println(err1);
    while (true) {
      ;
    }
  }
  
  // Initialize GPIO Pins for IMU round-robin
  // Init after IMU 1 setup to avoid double response on IMU_ADDRESS_1
  pinMode(ADDR_PIN_IMU_L, OUTPUT);
  pinMode(ADDR_PIN_IMU_R, OUTPUT);
  digitalWrite(ADDR_PIN_IMU_L, HIGH);
  digitalWrite(ADDR_PIN_IMU_R, LOW);

  // Init IMU 2
  int err2 = IMU_2.init(calib, IMU_ADDRESS_2);

  if (err2 != 0) {
    Serial.print("Error initializing IMU 2: ");
    Serial.println(err2);
    while (1) {
      ;
    }
  }

  // Init temperature sensor
  if (!mlx.begin()) {
    Serial.println("Error connecting to MLX sensor. Check wiring.");
    while (1);
  };

  // Init SD card
  if (!SD.begin(chipSelect)) {
    Serial.println("initialization failed. Things to check:");
    Serial.println("1. is a card inserted?");
    Serial.println("2. is your wiring correct?");
    Serial.println("3. did you change the chipSelect pin to match your shield or module?");
    Serial.println("Note: press reset button on the board and reopen this Serial Monitor after fixing your issue!");
    while (1);
  }

  String headerString = "Time (ms), ";
  headerString += "Acc_x C, Acc_y C, Acc_z C, Gyro_x C, Gyro_y C, Gyro_z C, IMU_temp C, ";
  headerString += "Acc_x L, Acc_y L, Acc_z L, Gyro_x L, Gyro_y L, Gyro_z L, IMU_temp L, ";
  headerString += "Acc_x R, Acc_y R, Acc_z R, Gyro_x R, Gyro_y R, Gyro_z R, IMU_temp R, ";
  headerString += "Obj_Temp (C), Ambient_Temp (C)";

  dataFile = SD.open("/datalog.txt", FILE_APPEND);

  // if the file is available, write to it:
  if (dataFile) {
    dataFile.println(headerString);
    dataFile.close();
    // print to the serial port too:
    Serial.println(headerString);
  }
  else {
    Serial.println("Error opening datalog file");
  }

  Serial.println("initialization done.");
  delay(5000);
}

void loop() {
  // open SD card file
  dataFile = SD.open("/datalog.txt", FILE_APPEND);
  if (!dataFile) {
    Serial.println("Error opening datalog file");
  }

  // record timestamp
  unsigned long mils = millis();
  dataPacket.mils = mils; // Into struct
  Serial.print("Time: "); // On serial
  Serial.print(mils); 
  // TODO: To SD Card
  dataFile.print(mils);
  dataFile.print(", ");

  // record center IMU
  activate_IMU_C();
  IMU_1.update();
  IMU_1.getAccel(&accelData);
  IMU_1.getGyro(&gyroData);
  IMU_temperature = IMU_1.getTemp();
  Serial.print("| IMU C: ");
  record_IMU_data(&dataPacket.imus[0]);

  // record left IMU
  activate_IMU_L();
  IMU_2.update();
  IMU_2.getAccel(&accelData);
  IMU_2.getGyro(&gyroData);
  IMU_temperature = IMU_2.getTemp();
  Serial.print(" | IMU L: ");
  record_IMU_data(&dataPacket.imus[1]);

  // record right IMU
  activate_IMU_R();
  IMU_2.update();
  IMU_2.getAccel(&accelData);
  IMU_2.getGyro(&gyroData);
  IMU_temperature = IMU_2.getTemp();
  Serial.print(" | IMU R: ");
  record_IMU_data(&dataPacket.imus[2]);

  // record temperature sensor
  double obj_temp = mlx.readObjectTempC();
  double ambient_temp = mlx.readAmbientTempC();
  dataPacket.t_obj = obj_temp; // Into struct
  dataPacket.t_amb = ambient_temp; // Into struct
  Serial.print(" | Temperature (C): "); // On Serial
  Serial.print(obj_temp);
  Serial.print("\t");
  Serial.println(ambient_temp);
  dataFile.print(obj_temp, 3); // To SD Card
  dataFile.print(", ");
  dataFile.println(ambient_temp, 3);

  // Write to and close SD Card file
  dataFile.close();

  // Send packet
  esp_now_send(broadcastAddress, (uint8_t *) &dataPacket, sizeof(dataPacket));

  delay(10);
}

void record_IMU_data(imu_record* const record) {
  // Into struct
  record->ax = accelData.accelX;
  record->ay = accelData.accelY;
  record->az = accelData.accelZ;
  record->gx = gyroData.gyroX;
  record->gy = gyroData.gyroY;
  record->gz = gyroData.gyroZ;
  record->temp = IMU_temperature;

  // To SD Card
  dataFile.print(accelData.accelX, 3);
  dataFile.print(", ");
  dataFile.print(accelData.accelY, 3);
  dataFile.print(", ");
  dataFile.print(accelData.accelZ, 3);
  dataFile.print(", ");
  dataFile.print(gyroData.gyroX, 3);
  dataFile.print(", ");
  dataFile.print(gyroData.gyroY, 3);
  dataFile.print(", ");
  dataFile.print(gyroData.gyroZ, 3);
  dataFile.print(", ");
  dataFile.print(IMU_temperature, 3);
  dataFile.print(", ");

  // On serial
  Serial.print(accelData.accelX);
  Serial.print("\t");
  Serial.print(accelData.accelY);
  Serial.print("\t");
  Serial.print(accelData.accelZ);
  Serial.print("\t");
  Serial.print(gyroData.gyroX);
  Serial.print("\t");
  Serial.print(gyroData.gyroY);
  Serial.print("\t");
  Serial.print(gyroData.gyroZ);
  Serial.print("\t");
  Serial.print(IMU_temperature);
}

// Takes L and R IMUs off the default address
void activate_IMU_C() {
  digitalWrite(ADDR_PIN_IMU_R, HIGH);
  digitalWrite(ADDR_PIN_IMU_L, HIGH);
}
// Puts L IMU on secondary address
void activate_IMU_L() {
  digitalWrite(ADDR_PIN_IMU_R, LOW);
  digitalWrite(ADDR_PIN_IMU_L, HIGH);
}
// Puts R IMU on secondary address
void activate_IMU_R() {
  digitalWrite(ADDR_PIN_IMU_L, LOW);
  digitalWrite(ADDR_PIN_IMU_R, HIGH);
}
