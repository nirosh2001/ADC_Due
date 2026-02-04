// #include <Arduino.h>
// #include <SPI.h>

// #include "registers.h"




// static inline void dvga_cs_low() { digitalWrite(DVGA_CS_PIN, LOW); }
// static inline void dvga_cs_high() { digitalWrite(DVGA_CS_PIN, HIGH); }

// static inline void le_low() { digitalWrite(LE, LOW); }
// static inline void le_high() { digitalWrite(LE, HIGH); }




// void att_write_reg(att_reg_address addr, uint8_t data)
// {
//   SPI.beginTransaction(SPISettings(100000, LSBFIRST, SPI_MODE0));
//   le_low();
//   SPI.transfer(data);
//   SPI.transfer(((uint8_t)addr)<<4); 
//   le_high();
//   le_low();
//   le_high();
//   Serial.print("reg addr 0x");
//   Serial.println(((uint8_t)addr)<<4, HEX);
//   Serial.print("data: 0x");
//   Serial.println(data, HEX);
//   SPI.endTransaction();
// }

// void att_init_serial()
// {
//   digitalWrite(LE, HIGH);  //latch enabled
//   digitalWrite(PBAR, HIGH); 
// }

// void att_serial(uint8_t g) 
// {
//   g &= 0x7F; // keep only 7 bits (0..127)

//   Serial.print("Setting gain to: ");
//   Serial.print(g);
//   Serial.print(" (binary: ");
//   for (int8_t i = 6; i >= 0; i--) Serial.print((g >> i) & 1);
//   Serial.println(")");
//   att_write_reg(att_reg_address::REG_ATTN, g);
  
// }

// void dvga_write_reg(uint8_t data)
// {
//   SPI.beginTransaction(SPISettings(100000, LSBFIRST, SPI_MODE0));
//   dvga_cs_high();
//   delay(10);
  
//   SPI.transfer(data);
//   SPI.transfer(data);
//   dvga_cs_low();
//   Serial.print("data: 0x");
//   Serial.println(data, HEX);
//   SPI.endTransaction();
// }

// void dvga_serial(uint8_t g) 
// {
//   g &= 0x3F; // keep only 6 bits (0..63)
//   g ^= 0x3F; 

//   Serial.print("Setting gain to: ");
//   Serial.print(g);
//   Serial.print(" (binary: ");
//   // for (int8_t i = 5; i >= 0; i--) Serial.print((g >> i) & 1);
//   // Serial.println(")");
//   dvga_write_reg(g);
  
// }

// void demod_init()
// {
//   demod_set_default();
//   demod_write_reg(demod_reg_address::ADDR_ATT_IP3IC, 0x04);
//   demod_write_reg(demod_reg_address::ADDR_BAND_LF1_CF2, 0xAA);
//   demod_write_reg(demod_reg_address::ADDR_LVCM_CF1,0x42);
//   demod_verify_all();
// }

// long readIntFromSerial(const char *prompt = nullptr) {
//   if (prompt) Serial.print(prompt);
//   String buf;
//   while (true) {
//     while (Serial.available()) {
//       char c = (char)Serial.read();
//       if (c == '\r' || c == '\n') {
//         if (buf.length() == 0) continue; // ignore empty line
//         long val = buf.toInt();
//         Serial.println();
//         return val;
//       } else if (c == 8 || c == 127) { // backspace/delete
//         if (buf.length() > 0) {
//           buf.remove(buf.length() - 1);
//           Serial.print("\b \b");
//         }
//       } else if (isPrintable(c)) {
//         buf += c;
//         Serial.print(c); // echo
//       }
//     }
//   }
// }

// void setup() {
//   // pinMode(ADC_CS_BAR, OUTPUT);
//   // pinMode(CONVST, OUTPUT);
//   // adc_cs_high();
//   // adc_convst_low();
  
//   pinMode(LE, OUTPUT);
//   pinMode(PBAR, OUTPUT);
//   digitalWrite(LE, HIGH);

//   pinMode(DEMOD_CS_PIN, OUTPUT);
//   pinMode(RFSW, OUTPUT);
//   digitalWrite(DEMOD_CS_PIN, HIGH);
//   digitalWrite(RFSW, LOW);

//   pinMode(DVGA_CS_PIN, OUTPUT);
//   pinMode(OE, OUTPUT);
//   pinMode(SENB, OUTPUT);
 
//   digitalWrite(DVGA_CS_PIN, LOW);
//   digitalWrite(OE, HIGH);    //enable output
//   digitalWrite(SENB, HIGH); //enable serial mode
//   SPI.begin();
  
//   Serial.begin(115200);
//   Serial.println("Starting...");
//   while (!Serial) ;
  
//   demod_init();
//   att_init_serial();
//   att_serial((uint8_t)1);
//   dvga_serial((uint8_t)63);
//   delay(1000);

// }

// void loop() {
  
//   // long userNumber = readIntFromSerial("Enter attenuation: ");
//   // Serial.print("You entered: ");
//   // Serial.println(userNumber);
  
  

//   // long userNumber = readIntFromSerial("Enter dvga gain: ");
//   // Serial.print("You entered: ");
//   // Serial.println(userNumber);
//   // dvga_serial((uint8_t)userNumber);
//   // delay(1000);
  
//   for (uint8_t g = 1; g <= 63; g++) {
//     dvga_serial(g);
//     delay(100);
//   }
//   // for (uint8_t g = 63; g > 1; g--) {
//   //   dvga_serial(g);
//   //   delay(500);
//   // }
//   // static int16_t buf[256];          // 256 samples = 512 bytes
//   // uint32_t next = micros();
//   // while (1) {
//   //   for (int i = 0; i < 256; i++) {
//   //     buf[i] = (int16_t)read_ads8867_once_fast();

//   //     next += 10;                   // 100 kSPS target
//   //     while ((int32_t)(micros() - next) < 0) {}
//   //   }

//   //   // Send one big packet (much faster than 256 small writes)
//   //   SerialUSB.write((uint8_t*)buf, sizeof(buf));
//   // }

// }
