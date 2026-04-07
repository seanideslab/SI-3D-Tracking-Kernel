/**
 * mcu_tracker.h
 * MCU 核心追蹤演算法標頭檔
 */
#ifndef MCU_TRACKER_H
#define MCU_TRACKER_H

#include <stdint.h>

/* 追蹤器狀態機枚舉 */
typedef enum {
    TRACKER_INIT        = 0,
    TRACKER_TRACKING    = 1,
    TRACKER_VISUAL_LOSS = 2,
    TRACKER_RELOCK      = 3,
} TrackerState;

/* 初始化設定 */
typedef struct {
    float q_pos;
    float q_vel;
    float r_pos;
} TrackerConfig;

/* 完整 Tracker 結構（在 .h 中定義以允許靜態配置） */
#define STATE_DIM_H     6
#define MEAS_DIM_H      3
#define IMM_MODEL_N_H   2
#define REPLAY_BUF_LEN_H 32

typedef struct {
    float x[STATE_DIM_H];
    float P[STATE_DIM_H][STATE_DIM_H];
} GaussState_t;

typedef struct {
    float    x[STATE_DIM_H];
    uint32_t timestamp_ms;
} ReplayEntry_t;

typedef struct {
    TrackerState   state;
    uint16_t       dropout_cnt;
    uint16_t       lock_cnt;

    GaussState_t   ekf;
    GaussState_t   imm[IMM_MODEL_N_H];
    float          imm_mu[IMM_MODEL_N_H];

    ReplayEntry_t  replay[REPLAY_BUF_LEN_H];
    uint8_t        rb_head;
    uint8_t        rb_len;

    float          out_pos[3];
    float          out_vel[3];
    float          rmse;

    float          Q_diag[STATE_DIM_H];
    float          R_diag[MEAS_DIM_H];
} Tracker;

/* API */
void tracker_init(Tracker *tk, const TrackerConfig *cfg);
void tracker_step(Tracker *tk, const float z[3], float dt,
                  uint32_t now_ms, uint32_t delay_ms);

#endif /* MCU_TRACKER_H */
