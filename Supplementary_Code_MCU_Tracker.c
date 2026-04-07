/**
 * mcu_tracker.c
 * MCU 核心目標追蹤演算法
 * 架構：狀態機切換 MSC-EKF（初始化/視覺遺失）與 IMM-UKF（穩定追蹤）
 * 適用平台：Cortex-M4/M7
 */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "mcu_tracker.h"

/* ─────────────── 常數 ─────────────── */
#define STATE_DIM       6
#define MEAS_DIM        3
#define IMM_MODEL_N     2
#define REPLAY_BUF_LEN  32
#define LOCK_THRESH_M   0.5f
#define DROPOUT_THRESH  5

static const float IMM_TRANS[IMM_MODEL_N][IMM_MODEL_N] = {
    {0.95f, 0.05f},
    {0.10f, 0.90f},
};

/* ─────────────── 3×3 逆矩陣 ─────────────── */
static int mat3_inv(const float A[3][3], float inv[3][3]) {
    float det =
        A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1]) -
        A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]) +
        A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
    if (fabsf(det) < 1e-9f) return -1;
    float id = 1.0f / det;
    inv[0][0] = id*(A[1][1]*A[2][2]-A[1][2]*A[2][1]);
    inv[0][1] = id*(A[0][2]*A[2][1]-A[0][1]*A[2][2]);
    inv[0][2] = id*(A[0][1]*A[1][2]-A[0][2]*A[1][1]);
    inv[1][0] = id*(A[1][2]*A[2][0]-A[1][0]*A[2][2]);
    inv[1][1] = id*(A[0][0]*A[2][2]-A[0][2]*A[2][0]);
    inv[1][2] = id*(A[0][2]*A[1][0]-A[0][0]*A[1][2]);
    inv[2][0] = id*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
    inv[2][1] = id*(A[0][1]*A[2][0]-A[0][0]*A[2][1]);
    inv[2][2] = id*(A[0][0]*A[1][1]-A[0][1]*A[1][0]);
    return 0;
}

/* ─────────────── EKF 預測（常速模型）─────────────── */
static void ekf_predict(GaussState_t *s, float dt, const float Q_diag[STATE_DIM]) {
    for (int i = 0; i < 3; i++) s->x[i] += s->x[i+3] * dt;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < STATE_DIM; j++) s->P[i][j] += dt * s->P[i+3][j];
        for (int j = 0; j < STATE_DIM; j++) s->P[j][i] += dt * s->P[j][i+3];
    }
    for (int i = 0; i < STATE_DIM; i++) s->P[i][i] += Q_diag[i];
}

/* ─────────────── EKF 更新（位置量測）─────────────── */
static void ekf_update(GaussState_t *s, const float z[3], const float R_diag[MEAS_DIM]) {
    float y[3] = {z[0]-s->x[0], z[1]-s->x[1], z[2]-s->x[2]};
    float S[3][3], Sinv[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            S[i][j] = s->P[i][j] + (i==j ? R_diag[i] : 0.0f);
    if (mat3_inv(S, Sinv) != 0) return;

    float K[STATE_DIM][3];
    for (int i = 0; i < STATE_DIM; i++)
        for (int j = 0; j < 3; j++) {
            K[i][j] = 0;
            for (int k = 0; k < 3; k++) K[i][j] += s->P[i][k] * Sinv[k][j];
        }
    for (int i = 0; i < STATE_DIM; i++)
        for (int k = 0; k < 3; k++) s->x[i] += K[i][k] * y[k];
    for (int i = 0; i < STATE_DIM; i++) {
        for (int j = 0; j < 3; j++) s->P[i][j] -= K[i][j];
        if (s->P[i][i] < 1e-4f) s->P[i][i] = 1e-4f;
    }
}

/* ─────────────── IMM 預測 ─────────────── */
static void imm_predict(Tracker *tk, float dt) {
    float c[IMM_MODEL_N], c_sum = 0.0f;
    for (int j = 0; j < IMM_MODEL_N; j++) {
        c[j] = 0;
        for (int i = 0; i < IMM_MODEL_N; i++) c[j] += IMM_TRANS[i][j] * tk->imm_mu[i];
        c_sum += c[j];
    }
    float mu_new[IMM_MODEL_N];
    for (int j = 0; j < IMM_MODEL_N; j++) mu_new[j] = c[j] / c_sum;

    float Q_cv[STATE_DIM], Q_ca[STATE_DIM];
    for (int i = 0; i < STATE_DIM; i++) { Q_cv[i] = tk->Q_diag[i]; Q_ca[i] = tk->Q_diag[i]*4.0f; }
    ekf_predict(&tk->imm[0], dt, Q_cv);
    ekf_predict(&tk->imm[1], dt, Q_ca);
    memcpy(tk->imm_mu, mu_new, sizeof(mu_new));
}

/* ─────────────── IMM 更新 + 融合 ─────────────── */
static void imm_update(Tracker *tk, const float z[3]) {
    float w[IMM_MODEL_N], w_sum = 0.0f;
    for (int j = 0; j < IMM_MODEL_N; j++) {
        ekf_update(&tk->imm[j], z, tk->R_diag);
        float d2 = 0;
        for (int i = 0; i < 3; i++) { float e = z[i]-tk->imm[j].x[i]; d2 += e*e; }
        w[j] = tk->imm_mu[j] * expf(-0.5f * d2);
        w_sum += w[j];
    }
    if (w_sum < 1e-12f) return;
    for (int j = 0; j < IMM_MODEL_N; j++) tk->imm_mu[j] = w[j] / w_sum;
    for (int i = 0; i < STATE_DIM; i++) {
        tk->ekf.x[i] = 0;
        for (int j = 0; j < IMM_MODEL_N; j++) tk->ekf.x[i] += tk->imm_mu[j] * tk->imm[j].x[i];
    }
}

/* ─────────────── Replay Buffer ─────────────── */
static void replay_push(Tracker *tk, uint32_t ts_ms) {
    uint8_t idx = (tk->rb_head + tk->rb_len) % REPLAY_BUF_LEN;
    memcpy(tk->replay[idx].x, tk->ekf.x, sizeof(float)*STATE_DIM);
    tk->replay[idx].timestamp_ms = ts_ms;
    if (tk->rb_len < REPLAY_BUF_LEN) tk->rb_len++;
    else tk->rb_head = (tk->rb_head + 1) % REPLAY_BUF_LEN;
}

static void replay_compensate(Tracker *tk, uint32_t now_ms, uint32_t delay_ms,
                               float out[STATE_DIM]) {
    uint32_t target = (now_ms > delay_ms) ? (now_ms - delay_ms) : 0;
    int best = tk->rb_head;
    uint32_t best_dt = UINT32_MAX;
    for (int i = 0; i < tk->rb_len; i++) {
        uint8_t idx = (tk->rb_head + i) % REPLAY_BUF_LEN;
        uint32_t t = tk->replay[idx].timestamp_ms;
        uint32_t d = (t <= target) ? (target-t) : (t-target);
        if (d < best_dt) { best_dt = d; best = idx; }
    }
    memcpy(out, tk->replay[best].x, sizeof(float)*STATE_DIM);
    float fwd = (float)(now_ms - tk->replay[best].timestamp_ms) * 1e-3f;
    for (int i = 0; i < 3; i++) out[i] += out[i+3] * fwd;
}

/* ─────────────── 狀態機 ─────────────── */
static void state_transition(Tracker *tk, int has_meas, float innov_norm) {
    switch (tk->state) {
    case TRACKER_INIT:
        if (has_meas && innov_norm < LOCK_THRESH_M) {
            if (++tk->lock_cnt >= 3) {
                memcpy(tk->imm[0].x, tk->ekf.x, sizeof(tk->ekf.x));
                memcpy(tk->imm[1].x, tk->ekf.x, sizeof(tk->ekf.x));
                tk->imm_mu[0]=0.7f; tk->imm_mu[1]=0.3f;
                tk->state = TRACKER_TRACKING; tk->lock_cnt = 0;
            }
        } else { tk->lock_cnt = 0; }
        break;
    case TRACKER_TRACKING:
        if (!has_meas) {
            if (++tk->dropout_cnt >= DROPOUT_THRESH) {
                tk->state = TRACKER_VISUAL_LOSS; tk->dropout_cnt = 0;
            }
        } else { tk->dropout_cnt = 0; }
        break;
    case TRACKER_VISUAL_LOSS:
        if (has_meas) {
            if (++tk->lock_cnt >= 2) { tk->state = TRACKER_RELOCK; tk->lock_cnt = 0; }
        } else { tk->lock_cnt = 0; }
        break;
    case TRACKER_RELOCK:
        if (innov_norm < LOCK_THRESH_M) tk->state = TRACKER_TRACKING;
        else if (!has_meas)             tk->state = TRACKER_VISUAL_LOSS;
        break;
    }
}

/* ─────────────── 公開 API ─────────────── */
void tracker_init(Tracker *tk, const TrackerConfig *cfg) {
    memset(tk, 0, sizeof(Tracker));
    tk->state = TRACKER_INIT;
    tk->imm_mu[0] = 0.7f; tk->imm_mu[1] = 0.3f;
    for (int i = 0; i < 3; i++) {
        tk->Q_diag[i]   = cfg ? cfg->q_pos : 0.01f;
        tk->Q_diag[i+3] = cfg ? cfg->q_vel : 0.10f;
        tk->R_diag[i]   = cfg ? cfg->r_pos : 0.05f;
    }
    for (int i = 0; i < STATE_DIM; i++) {
        tk->ekf.P[i][i]    = 1.0f;
        tk->imm[0].P[i][i] = 1.0f;
        tk->imm[1].P[i][i] = 1.0f;
    }
}

void tracker_step(Tracker *tk, const float z[3], float dt,
                  uint32_t now_ms, uint32_t delay_ms) {
    int has_meas = (z != NULL);

    /* 預測 */
    if (tk->state == TRACKER_TRACKING || tk->state == TRACKER_RELOCK)
        imm_predict(tk, dt);
    else
        ekf_predict(&tk->ekf, dt, tk->Q_diag);

    /* 更新 */
    float innov_norm = 0.0f;
    if (has_meas) {
        float ref[STATE_DIM];
        if (delay_ms > 0) {
            replay_compensate(tk, now_ms, delay_ms, ref);
            for (int i = 0; i < 3; i++) { float e = z[i]-ref[i]; innov_norm += e*e; }
        } else {
            for (int i = 0; i < 3; i++) { float e = z[i]-tk->ekf.x[i]; innov_norm += e*e; }
        }
        innov_norm = sqrtf(innov_norm);

        if (tk->state == TRACKER_TRACKING)
            imm_update(tk, z);
        else
            ekf_update(&tk->ekf, z, tk->R_diag);

        tk->rmse = 0.95f * tk->rmse + 0.05f * innov_norm;
    }

    state_transition(tk, has_meas, innov_norm);
    replay_push(tk, now_ms);

    for (int i = 0; i < 3; i++) { tk->out_pos[i] = tk->ekf.x[i]; tk->out_vel[i] = tk->ekf.x[i+3]; }
}

/* ─────────────── Demo（桌面驗證用）─────────────── */
#ifdef MCU_TRACKER_DEMO
#include <stdio.h>
int main(void) {
    Tracker tk;
    TrackerConfig cfg = {0.01f, 0.1f, 0.05f};
    tracker_init(&tk, &cfg);
    uint32_t t_ms = 0;
    float true_x = 0.0f;
    for (int frame = 0; frame < 50; frame++) {
        true_x += 0.1f;
        float z[3] = {true_x + 0.03f*(float)((frame%3)-1), 0.0f, 0.0f};
        int dropout = (frame >= 20 && frame < 30);
        tracker_step(&tk, dropout ? NULL : z, 0.02f, t_ms, 0);
        t_ms += 20;
        const char *sname[] = {"INIT","TRACKING","VISUAL_LOSS","RELOCK"};
        printf("f=%02d %-12s pos=(%.3f,%.3f,%.3f) rmse=%.4f\n",
            frame, sname[tk.state], tk.out_pos[0], tk.out_pos[1], tk.out_pos[2], tk.rmse);
    }
    return 0;
}
#endif
