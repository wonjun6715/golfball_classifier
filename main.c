#include <AFMotor.h>
#include <Servo.h>
#include <ArduinoJson.h>
#include <LiquidCrystal_I2C.h>

AF_Stepper motor_step(200,1);
AF_Stepper motor_act(60,2);
int flag = 0;
Servo myservo;
StaticJsonDocument<20> doc;

void setup() {
  lcd.init();
  lcd.backlight();
  motor_step.setSpeed(8);
  motor_step.onestep(FORWARD, DOUBLE);
  motor_step.release();

  motor_act.setSpeed(160);  //스텝모터용 무거운거는 15
  motor_act.onestep(FORWARD, INTERLEAVE);
  motor_act.release();

  Serial.begin(9600);
  myservo.attach(10);

  delay(1000);
  
}

void doStep(){
  motor_step.step(40,FORWARD,DOUBLE);
  doc["STEP"] = 1;
  serializeJson(doc, Serial);
  Serial.println();
}

void doAction(){
  while(Serial.available() == 0);

  deserializeJson(doc, Serial);

  motor_act.step(900, FORWARD, INTERLEAVE);
  delay(50);
  motor_act.step(900, BACKWARD, INTERLEAVE);

  int act_flag = doc["act"];
  float accuracy = doc["accuracy"];
  // 수정 필요
  if(act_flag == 1){ // normal ball
    lcd.setCursor(0,0);
    lcd.print("NORMAL");
    lcd.setCursor(0,1);
    lcd.print(accuracy, 2);
    myservo.write(80);
    delay(3000);
    myservo.write(90);
  } else if(act_flag == 2){ // broken ball
    lcd.setCursor(2,0);
    lcd.print("BROKEN");
    lcd.setCursor(0,1);
    lcd.print(accuracy, 2);
    myservo.write(100);
    delay(3000);
    myservo.write(90);
  }
}
void loop() {
  doc.clear();
  doStep();
  doAction();
  
}
  