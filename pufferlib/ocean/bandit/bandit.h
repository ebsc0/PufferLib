#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#define MAX_HISTORY 1024

typedef struct {
    float score;
    float n; // Required as last field for logging
} Log;

typedef struct {
    Log log;                     // Required field
    float* observations;         // Required field
    int* actions;                // Required field
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field

    int k;                       // Number of arms
    float* means;                // Reward probability per arm
    float total_reward;          // Sum of rewards
    float history[MAX_HISTORY];  // Reward history for rendering
    int step;                    // Steps elapsed
} Bandit;

static inline float randf(){return (float)rand()/ (float)RAND_MAX;}

void init(Bandit* env){
    env->means = (float*)calloc(env->k, sizeof(float));
    for(int i=0;i<env->k;i++) env->means[i] = 0.5f;
    env->total_reward = 0.0f;
    env->step = 0;
    memset(env->history, 0, sizeof(env->history));
}

void c_reset(Bandit* env){
    env->log = (Log){0};
    for(int i=0;i<env->k;i++) env->means[i] = 0.5f;
    env->total_reward = 0.0f;
    env->step = 0;
    env->observations[0] = 0.0f;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    memset(env->history, 0, sizeof(env->history));
}

void c_step(Bandit* env){
    int action = env->actions[0];
    if(action < 0) action = 0;
    if(action >= env->k) action = env->k - 1;

    for(int i=0;i<env->k;i++){
        env->means[i] += (randf() - 0.5f) * 0.1f; // Drift
        if(env->means[i] < 0.0f) env->means[i] = 0.0f;
        if(env->means[i] > 1.0f) env->means[i] = 1.0f;
    }

    env->rewards[0] = randf() < env->means[action] ? 1.0f : 0.0f;
    env->terminals[0] = 0;
    env->total_reward += env->rewards[0];
    if(env->step < MAX_HISTORY)
        env->history[env->step] = env->total_reward;
    env->log.score += env->rewards[0];
    env->log.n += 1.0f;
    env->step += 1;
    env->observations[0] = 0.0f;
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

void c_render(Bandit* env){
    if(!IsWindowReady()){
        InitWindow(600, 400, "PufferLib Bandit");
        SetTargetFPS(60);
    }
    if(IsKeyDown(KEY_ESCAPE)) exit(0);

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    int width = GetScreenWidth();
    int height = GetScreenHeight();
    int count = env->step < MAX_HISTORY ? env->step : MAX_HISTORY;

    float max_val = 1.0f;
    for(int i=0;i<count;i++)
        if(env->history[i] > max_val) max_val = env->history[i];

    for(int i=1;i<count;i++){
        float x1 = (i-1) * (float)width / (MAX_HISTORY-1);
        float y1 = height - env->history[i-1]/max_val * height;
        float x2 = i * (float)width / (MAX_HISTORY-1);
        float y2 = height - env->history[i]/max_val * height;
        DrawLine((int)x1, (int)y1, (int)x2, (int)y2, PUFF_CYAN);
    }

    DrawText(TextFormat("Total Reward: %.1f", env->total_reward), 20, 20, 20, PUFF_WHITE);
    EndDrawing();
}

void c_close(Bandit* env){
    if(IsWindowReady()) CloseWindow();
    free(env->means);
}

