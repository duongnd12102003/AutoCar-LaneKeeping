#define CAMERA_MODEL_ESP32S3_EYE
#if defined(CAMERA_MODEL_AI_THINKER)
  #define PIN_MOTOR_A_PWM 12
  #define PIN_MOTOR_B_PWM 13
#elif defined(CAMERA_MODEL_ESP32S3_EYE)
  #define PIN_MOTOR_A_PWM 41
  #define PIN_MOTOR_B_PWM 42
#endif

#define DEBUG_MODE

#include "DCMotorControl.h"
#include "camera_pins.h"
#include "driver/uart.h"
#include "esp_camera.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <WiFi.h>
#include <WiFiUdp.h>
#include <cstring>
#include <stdio.h>
#include <stdlib.h> 

const char *ssid = "ACE_UDP";
const char *password = "12345678";

WiFiUDP Udp;
const char *udpAddressLaptop = "192.168.1.114";
const uint16_t udpPortLaptop = 3000;
const uint16_t udpPortCam = 3001;
static camera_config_t camera_config = {
    .pin_pwdn = PWDN_GPIO_NUM,
    .pin_reset = RESET_GPIO_NUM,
    .pin_xclk = XCLK_GPIO_NUM,
    .pin_sscb_sda = SIOD_GPIO_NUM,
    .pin_sscb_scl = SIOC_GPIO_NUM,
    .pin_d7 = Y9_GPIO_NUM,
    .pin_d6 = Y8_GPIO_NUM,
    .pin_d5 = Y7_GPIO_NUM,
    .pin_d4 = Y6_GPIO_NUM,
    .pin_d3 = Y5_GPIO_NUM,
    .pin_d2 = Y4_GPIO_NUM,
    .pin_d1 = Y3_GPIO_NUM,
    .pin_d0 = Y2_GPIO_NUM,
    .pin_vsync = VSYNC_GPIO_NUM,
    .pin_href = HREF_GPIO_NUM,
    .pin_pclk = PCLK_GPIO_NUM,
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_QVGA,
    .jpeg_quality = 10,
    .fb_count = 1,
};

#if defined(DEBUG_MODE)
void uart_init() {
    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE};
    uart_param_config(UART_NUM_1, &uart_config);
    uart_set_pin(UART_NUM_1, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    uart_driver_install(UART_NUM_1, 256, 0, 0, NULL, 0);
}
#endif

esp_err_t camera_init() {
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        printf("Camera Init Failed\n");
        return err;
    }
    sensor_t *pSensor = esp_camera_sensor_get();
    pSensor->set_vflip(pSensor, 0);
    pSensor->set_hmirror(pSensor, 0);
    printf("Camera Init OK\n");
    return ESP_OK;
}

void wifi_init() {
    WiFi.begin(ssid, password);
    Serial.println("");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    printf("Connected to WiFi\n");
    Serial.print("ESP32 IP: ");
    Serial.println(WiFi.localIP());
    Serial.print("ESP32 MAC: ");
    Serial.println(WiFi.macAddress());

    Udp.begin(udpPortCam);
    printf("Now listening at IP %s port %d\n", WiFi.localIP().toString().c_str(), udpPortCam);
}

void sendImageTask(void *pvParameters) {
    camera_fb_t *fb = NULL;
    printf("Send image\n");
    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            printf("Camera Capture Failed\n");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        Udp.beginPacket(udpAddressLaptop, udpPortLaptop);
        Udp.write(fb->buf, fb->len);
        Udp.endPacket();
        esp_camera_fb_return(fb);
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

static inline void trim_line(char *buf) {
    size_t L = strlen(buf);
    while (L && (buf[L-1] == '\n' || buf[L-1] == '\r' || buf[L-1] == ' ')) {
        buf[--L] = '\0';
    }
}

void receiveMessageTask(void *pvParameters) {
    DCMotorControl motorControl(PIN_MOTOR_A_PWM, PIN_MOTOR_B_PWM);

    motorControl.setInvertSteer(true);
    motorControl.setSideGains(1.00f, 1.00f);
    motorControl.setTrim(+0, +0);
    uint8_t packetSize;
    char MovementCmd[32]; 
    printf("Listen control\n");
    while (true) {
        packetSize = Udp.parsePacket();
        if (packetSize) {
            int len = Udp.read(MovementCmd, sizeof(MovementCmd) - 1);
            if (len > 0) {
                MovementCmd[len] = '\0';
                trim_line(MovementCmd);
                char *spacePos1 = strchr(MovementCmd, ' ');
                char *spacePos2 = nullptr;
                if (spacePos1) spacePos2 = strchr(spacePos1 + 1, ' ');

                if (spacePos1 && spacePos2) {
                    *spacePos1 = '\0';
                    *spacePos2 = '\0';
                    char *directionStr = MovementCmd;
                    char *speedStr     = spacePos1 + 1;
                    char *alphaStr     = spacePos2 + 1;
                    while (*speedStr == ' ')  speedStr++;
                    while (*alphaStr == ' ')  alphaStr++;

                    uint8_t unDirection = (uint8_t)atoi(directionStr);
                    uint8_t unSpeed     = (uint8_t)atoi(speedStr);
                    int8_t  nAlpha      = (int8_t)atoi(alphaStr);

                    if (unSpeed > 255) unSpeed = 255;
                    if (nAlpha > 127)  nAlpha = 127;
                    if (nAlpha < -128) nAlpha = -128;

                    printf("Cmd -> Dir:%u Speed:%u Alpha:%d\n", unDirection, unSpeed, nAlpha);
                    motorControl.CarMovementControl(unDirection, unSpeed, nAlpha);
                } else {
                    printf("Invalid command: %s\n", MovementCmd);
                }
            }
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

void setup() {
    #if defined(DEBUG_MODE)
      uart_init();
    #endif
    camera_init();
    wifi_init();

    xTaskCreatePinnedToCore(
        sendImageTask,
        "SendImageTask",
        8192,
        NULL,
        2,
        NULL,
        1);

    xTaskCreatePinnedToCore(
        receiveMessageTask,
        "RecvMsgTask",
        4096,
        NULL,
        1,
        NULL,
        0);
}

void loop() {
}
