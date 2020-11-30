#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <__clang_cuda_runtime_wrapper.h>
#include <algorithm>
#include <ctime>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <stdlib.h> /* srand, rand */
#include <time.h>   /* time */
#include <vector>

using namespace std;

__host__ int randomInt(int, int);
__host__ float randomFloat(float, float);

class Agent {
public:
  float contagionProba;        // 0.02 - 0.03
  float extContagionProba;     // 0.02 - 0.03
  float deathProba;            // 0.007 - 0.07
  float movProba;              // 0.3 - 0.5
  float shortDistanceMovProba; // 0.7 - 0.9
  int incubationTime;          // 5 - 6
  int recoveryTime;            // 14

  int status; /* (0) No infected, (1) infected, (-1) Quarantine (-3) Decease
               */
  float posX; // 0 - p
  float posY; // 0 - q

  __device__ __host__ void generate(float x, float y) {
    posX = x;
    posY = y;
    contagionProba = randomFloat(0.02, 0.03);
    extContagionProba = randomFloat(0.02, 0.03);
    deathProba = randomFloat(0.007, 0.07);
    movProba = randomFloat(0.3, 0.5);
    shortDistanceMovProba = randomFloat(0.7, 0.9);
    incubationTime = randomInt(5, 6);
    recoveryTime = 14;
    status = 0;
  }
};

struct Stats {
  int accumulateContagion;
  int newContagion;
  int accumulateRecovered;
  int newRecovered;
  int accumulateDeads;
  int newDeads;
  int firstContagion;
  int firstRecovered;
  int firstDead;
  int lastContagion;
  int lastRecovered;
  int lastDead;
  int halfContagion;
  int halfRecovered;
  int halfDead;
};
// Quantity of agents
const int numAgents = 10240;
// Days of duration of the simulation
const int numDays = 61;
// Maximum number of movements per day
const int maxNumMovDay = 30;
// Maximum radius of local movement
const int radiusMaxMovLocal = 5;

int xSize = 500, ySize = 500;

int deaths = 0;

int contagions = 0;

void initAgents(vector<Agent> &agents) {
  for (int i = 0; i < numAgents; i++) {
    int posX = 0;
    int posY = 0;

    posX = randomInt(0, 500);
    posY = randomInt(0, 500);

    Agent newAgente = Agent();

    newAgente.generate(posX, posY);
    agents.push_back(newAgente);
  }
}
float EuclideanDistance_CPU(Agent, Agent);
void simulate();
void contagion(Agent *, vector<Agent> &);
void movility(Agent *);
void externContagion(Agent *);
void contagionEffects(Agent *);
void decease(Agent *);

__device__ float generate(curandState *globalState, int ind) {
  curandState localState = globalState[ind];
  float RANDOM = curand_uniform(&localState);
  globalState[ind] = localState;
  return RANDOM;
}

__device__ float generateRand(curandState *globalState, int ind, float low,
                              float high) {
  curandState localState = globalState[ind];
  float RANDOM = curand_uniform(&localState) * (high);
  if (RANDOM < low) {
    RANDOM = low;
  }
  globalState[ind] = localState;
  return RANDOM;
}

__global__ void setup_kernel(curandState *state, unsigned long seed) {
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int threadId = blockId * (blockDim.x * blockDim.y) +
                 (threadIdx.y * blockDim.x) + threadIdx.x;
  curand_init(seed, threadId, 0, &state[threadId]);
}

__device__ float EuclideanDistance(Agent agent1, Agent Agent2) {
  float suma = 0;
  suma = pow(agent1.posX - Agent2.posX, 2);
  suma += pow(agent1.posY - Agent2.posY, 2);
  return float(sqrt(suma));
}
float EuclideanDistance_CPU(Agent agent1, Agent Agent2) {
  float suma = 0;
  suma = pow(agent1.posX - Agent2.posX, 2);
  suma += pow(agent1.posY - Agent2.posY, 2);
  return float(sqrt(suma));
}
__global__ void GPU_contagio(Agent *agent, curandState *globalState) {
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int thisId = blockId * (blockDim.x * blockDim.y) +
               (threadIdx.y * blockDim.x) + threadIdx.x;
  if (agent[thisId].status == 0) {
    for (int bIdY = 0; bIdY < gridDim.y; bIdY++) {
      for (int bIdX = 0; bIdX < gridDim.x; bIdX++) {
        for (int tIdY = 0; tIdY < blockDim.y; tIdY++) {
          for (int tIdX = 0; tIdX < blockDim.x; tIdX++) {
            int blockId = bIdX + bIdY * gridDim.x;
            int otherId = blockId * (blockDim.x * blockDim.y) +
                          (tIdY * blockDim.x) + tIdX;
            if (agent[otherId].status == 1) {
              if (EuclideanDistance(agent[thisId], agent[otherId]) <= 1.0) {
                if (generate(globalState, thisId) <=
                    agent[thisId].contagionProba) {
                  agent[thisId].status = 1;
                  return;
                }
              }
            }
          }
        }
      }
    }
  }
}

__global__ void GPU_Efects(Agent *agent, curandState *globalState) {
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int thisId = blockId * (blockDim.x * blockDim.y) +
               (threadIdx.y * blockDim.x) + threadIdx.x;

  if (agent[thisId].status == 1) {
    if (agent[thisId].incubationTime == 0) {
      agent[thisId].status = -1;
    }
    agent[thisId].incubationTime -= 1;

  } else if (agent[thisId].status == -1) {

    agent[thisId].recoveryTime--;
  }
}

__global__ void GPU_movility(Agent *agent, curandState *globalState,
                             int *maxMovementDistance) {
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int threadId = blockId * (blockDim.x * blockDim.y) +
                 (threadIdx.y * blockDim.x) + threadIdx.x;

  float xLimit = 500.0, yLimit = 500.0;

  float actualX = agent[threadId].posX, actualY = agent[threadId].posY;
  float movementX = 0.0, movementY = 0.0;
  bool movValido = false;
  if ((agent[threadId].status == -1 && agent[threadId].recoveryTime > 0) ||
      agent[threadId].status == -2) {
    return;
  }
  if (generate(globalState, threadId) <= agent[threadId].movProba) {
    if (generate(globalState, threadId) <=
        agent[threadId].shortDistanceMovProba) {
      do {
        movementX = 2 * generate(globalState, threadId) - 1;
        movementY = 2 * generate(globalState, threadId) - 1;
        if (movementX + actualX >= 0.0 || movementX + actualX < xLimit) {
          if (movementY + actualY >= 0.0 || movementY + actualY < yLimit) {
            if ((movementX * movementX) + (movementY * movementY) <=
                (*maxMovementDistance * *maxMovementDistance)) {
              movValido = true;
            }
          }
        }
      } while (!movValido);
    } else {
      do {
        movementX = xLimit * generate(globalState, threadId);
        movementY = yLimit * generate(globalState, threadId);
        if (movementX + actualX >= 0.0 || movementX + actualX < xLimit) {
          if (movementY + actualY >= 0.0 || movementY + actualY < yLimit) {
            if ((movementX * movementX) + (movementY * movementY) <=
                (*maxMovementDistance * *maxMovementDistance)) {
              movValido = true;
            }
          }
        }
      } while (!movValido);
    }
    agent[threadId].posX = actualX + movementX;
    agent[threadId].posY = actualY + movementY;
  }
}

__global__ void GPU_externContagion(Agent *agent, curandState *globalState) {
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int threadId = blockId * (blockDim.x * blockDim.y) +
                 (threadIdx.y * blockDim.x) + threadIdx.x;
  if (agent[threadId].status == 0) {
    if (generate(globalState, threadId) * 1.0 <=
        agent[threadId].extContagionProba) {
      agent[threadId].status = 1;
    }
  }
}
__global__ void GPU_Decease(Agent *agent, curandState *globalState) {
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int threadId = blockId * (blockDim.x * blockDim.y) +
                 (threadIdx.y * blockDim.x) + threadIdx.x;

  if (agent[threadId].status == -1 && agent[threadId].recoveryTime > 0) {
    float posibilidad = generate(globalState, threadId);
    if (posibilidad <= agent[threadId].deathProba) {
      agent[threadId].status = -2;
    }
  }
}

__global__ void initAgents(Agent *agent, curandState *globalState) {

  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int threadId = blockId * (blockDim.x * blockDim.y) +
                 (threadIdx.y * blockDim.x) + threadIdx.x;

  float posX = generate(globalState, threadId) * 500.0;
  float posY = generate(globalState, threadId) * 500.0;

  Agent agentNew = Agent();
  agentNew.posX = posX;
  agentNew.posY = posY;

  agentNew.contagionProba = generateRand(globalState, threadId, 0.02, 0.03);
  agentNew.extContagionProba = generateRand(globalState, threadId, 0.02, 0.03);
  agentNew.deathProba = generateRand(globalState, threadId, 0.007, 0.07);
  agentNew.movProba = generateRand(globalState, threadId, 0.3, 0.5);
  agentNew.shortDistanceMovProba =
      generateRand(globalState, threadId, 0.7, 0.9);
  /* Incubation time */
  agentNew.incubationTime = 5;
  if (generate(globalState, threadId) > 0.5) {
    agentNew.incubationTime = 6;
  }
  agentNew.recoveryTime = 14;
  agentNew.status = 0;
  agent[threadId] = agentNew;
}
/* Function to extract the stats from the GPU implementation */
__host__ void getStats(Agent *agentsCPU, Stats *stats, int day) {
  stats->newContagion = 0;
  stats->newRecovered = 0;
  stats->newDeads = 0;
  for (int i = 0; i < numAgents; i++) {

    if (agentsCPU[i].status != 0) {
      // Contagios

      stats->newContagion++;
      if (stats->accumulateContagion == 0) {
        stats->firstContagion = day;
      } else if (stats->accumulateContagion == numAgents / 2) {
        stats->halfContagion = day;
      }
      stats->lastContagion = day;
    }
    if (agentsCPU[i].status == -1 && agentsCPU[i].recoveryTime <= 0) {
      // Recovered

      stats->newRecovered++;
      if (stats->accumulateRecovered == 0) {
        stats->firstRecovered = day;
      } else if (stats->accumulateRecovered == numAgents / 2) {
        stats->halfRecovered = day;
      }
      stats->lastRecovered = day;

    } else if (agentsCPU[i].status == -2) {
      // Deaths

      stats->newDeads++;

      if (stats->accumulateDeads == 0) {
        stats->firstDead = day;
      } else if (stats->accumulateDeads == numAgents / 2) {
        stats->halfDead = day;
      }
      stats->lastDead = day;
    }
  }

  stats->newRecovered -= stats->accumulateRecovered;
  stats->accumulateRecovered += stats->newRecovered;
  stats->newContagion -= stats->accumulateContagion;
  stats->accumulateContagion += stats->newContagion;

  stats->newDeads -= stats->accumulateDeads;
  stats->accumulateDeads += stats->newDeads;
  ;
}
/* Function to extract the stats from the CPU implementation */
__host__ void getStats_CPU(vector<Agent> &agents, Stats *stats, int day) {
  stats->newContagion = 0;
  stats->newRecovered = 0;
  stats->newDeads = 0;
  for (auto &agent : agents) {

    if (agent.status != 0) {
      // Contagios

      stats->newContagion++;
      if (stats->accumulateContagion == 0) {
        stats->firstContagion = day;
      } else if (stats->accumulateContagion == numAgents / 2) {
        stats->halfContagion = day;
      }
      stats->lastContagion = day;
    }
    if (agent.status == -1 && agent.recoveryTime <= 0) {
      // Recovered

      stats->newRecovered++;
      if (stats->accumulateRecovered == 0) {
        stats->firstRecovered = day;
      } else if (stats->accumulateRecovered == numAgents / 2) {
        stats->halfRecovered = day;
      }
      stats->lastRecovered = day;

    } else if (agent.status == -2) {
      // Deaths

      stats->newDeads++;

      if (stats->accumulateDeads == 0) {
        stats->firstDead = day;
      } else if (stats->accumulateDeads == numAgents / 2) {
        stats->halfDead = day;
      }
      stats->lastDead = day;
    }
  }

  stats->newRecovered -= stats->accumulateRecovered;
  stats->accumulateRecovered += stats->newRecovered;
  stats->newContagion -= stats->accumulateContagion;
  stats->accumulateContagion += stats->newContagion;

  stats->newDeads -= stats->accumulateDeads;
  stats->accumulateDeads += stats->newDeads;
  ;
}

int main() {

  // ++++++++++++++++++++++++++++++++ CPU ++++++++++++++++++++++++++++++++++++
  printf("++++++++++++++++++++++++++++++++ CPU "
         "++++++++++++++++++++++++++++++++++++\n") srand(time(NULL));

  simulate();

  // ++++++++++++++++++++++++++++++++ GPU ++++++++++++++++++++++++++++++++++++
  printf("\n\n++++++++++++++++++++++++++++++++ GPU "
         "++++++++++++++++++++++++++++++++++++\n") srand(time(NULL));
  // Variables Creation
  static Agent agents[numAgents];

  Agent *agentsGPU;
  Agent *agentsCPU;
  int seed = rand();
  float gpu_time_enlapsed;

  float gpu_enlapsed_time_counter = 0;
  int *devRadiusMaxMovLocal;

  // Memory Allocation
  const size_t size = size_t(numAgents) * sizeof(Agent);
  agentsCPU = (Agent *)malloc(size);
  cudaMalloc((void **)&agentsGPU, size);
  cudaMemcpy(agentsGPU, &agents[0], size, cudaMemcpyHostToDevice);
  curandState *devStates;
  cudaMalloc(&devStates, numAgents * sizeof(curandState));
  cudaMalloc(&devRadiusMaxMovLocal, sizeof(int));
  cudaMemcpy(devRadiusMaxMovLocal, &radiusMaxMovLocal, sizeof(int),
             cudaMemcpyHostToDevice);

  // Kernel's configuration
  dim3 block(5, 2);
  dim3 grid(32, 32);
  srand(time(0));

  // Start GPU timers
  cudaEvent_t start_GPU, end_GPU;
  cudaEventCreate(&start_GPU);
  cudaEventCreate(&end_GPU);

  // setup the kernel for the random numbers
  cudaEventRecord(start_GPU, 0);
  setup_kernel<<<grid, block>>>(devStates, time(NULL));
  cudaEventRecord(end_GPU, 0);
  cudaEventSynchronize(end_GPU);
  cudaEventElapsedTime(&gpu_time_enlapsed, start_GPU, end_GPU);
  gpu_enlapsed_time_counter += gpu_time_enlapsed;
  cudaEventDestroy(start_GPU);
  cudaEventDestroy(end_GPU);

  cudaEventRecord(start_GPU, 0);
  initAgents<<<grid, block>>>(agentsGPU, devStates);
  cudaMemcpy(agentsCPU, agentsGPU, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(end_GPU, 0);
  cudaEventSynchronize(end_GPU);
  cudaEventElapsedTime(&gpu_time_enlapsed, start_GPU, end_GPU);
  gpu_enlapsed_time_counter += gpu_time_enlapsed;
  cudaEventDestroy(start_GPU);
  cudaEventDestroy(end_GPU);

  int day = 0, movement;
  Stats currentStats = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Stats *currentStatsPointer = &currentStats;
  while (day < numDays) {
    movement = 0;
    // Extern Contagion GPU implementation
    cudaEventRecord(start_GPU, 0);
    GPU_externContagion<<<grid, block>>>(agentsGPU, devStates);
    cudaDeviceSynchronize();
    cudaEventRecord(end_GPU, 0);
    cudaEventSynchronize(end_GPU);
    cudaEventElapsedTime(&gpu_time_enlapsed, start_GPU, end_GPU);
    gpu_enlapsed_time_counter += gpu_time_enlapsed;
    cudaEventDestroy(start_GPU);
    cudaEventDestroy(end_GPU);

    while (movement < maxNumMovDay) {
      // Contagion GPU Implementation
      cudaEventRecord(start_GPU, 0);
      GPU_contagio<<<grid, block>>>(agentsGPU, devStates);
      cudaDeviceSynchronize();
      cudaEventRecord(end_GPU, 0);
      cudaEventSynchronize(end_GPU);
      cudaEventElapsedTime(&gpu_time_enlapsed, start_GPU, end_GPU);
      gpu_enlapsed_time_counter += gpu_time_enlapsed;
      cudaEventDestroy(start_GPU);
      cudaEventDestroy(end_GPU);

      // Agents Movility GPU Implementation
      cudaEventRecord(start_GPU, 0);
      GPU_movility<<<grid, block>>>(agentsGPU, devStates, devRadiusMaxMovLocal);
      cudaDeviceSynchronize();
      cudaEventRecord(end_GPU, 0);
      cudaEventSynchronize(end_GPU);
      cudaEventElapsedTime(&gpu_time_enlapsed, start_GPU, end_GPU);
      gpu_enlapsed_time_counter += gpu_time_enlapsed;
      cudaEventDestroy(start_GPU);
      cudaEventDestroy(end_GPU);
      movement++;
    }

    // Infecction Effects GPU implementation
    cudaEventRecord(start_GPU, 0);
    GPU_Efects<<<grid, block>>>(agentsGPU, devStates);
    cudaDeviceSynchronize();
    cudaEventRecord(end_GPU, 0);
    cudaEventSynchronize(end_GPU);
    cudaEventElapsedTime(&gpu_time_enlapsed, start_GPU, end_GPU);
    gpu_enlapsed_time_counter += gpu_time_enlapsed;
    cudaEventDestroy(start_GPU);
    cudaEventDestroy(end_GPU);

    // Agents Decease GPU implementation
    cudaEventRecord(start_GPU, 0);
    GPU_Decease<<<grid, block>>>(agentsGPU, devStates);
    cudaDeviceSynchronize();
    cudaMemcpy(agentsCPU, agentsGPU, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(end_GPU, 0);
    cudaEventSynchronize(end_GPU);
    cudaEventElapsedTime(&gpu_time_enlapsed, start_GPU, end_GPU);
    gpu_enlapsed_time_counter += gpu_time_enlapsed;
    cudaEventDestroy(start_GPU);
    cudaEventDestroy(end_GPU);

    // Print Simulation Stats Every Day
    getStats(agentsCPU, currentStatsPointer, day);
    printf("Day: %d, New infected: %d, New  Recovered: %d, New "
           "Dead: %d.\n",
           day, currentStats.newContagion, currentStats.newRecovered,
           currentStats.newDeads);
    if (day % 5 == 0 && day != 0) {
      printf("===== Day: %d, Accumaled infected: %d, Accumaled  Recovered:%d,"
             "Accumaled "
             "Dead: %d. =====\n",
             day, currentStats.accumulateContagion,
             currentStats.accumulateRecovered, currentStats.accumulateDeads);
    }

    day++;
  }

  // Print GPU Timers
  printf(" ------ Time Enlapsed in GPU Implementation: %.2f ms. -----\n",
         gpu_enlapsed_time_counter);

  // Print Final Stats
  getStats(agentsCPU, &currentStats, day);
  printf("======== First Contagion %d, Half Contagion %d, Last Contagion %d "
         "========\n",
         currentStats.firstContagion, currentStats.halfContagion,
         currentStats.lastContagion);
  printf("======== First Recovered %d,Half Recovered %d, Last Recovered %d "
         "========\n",
         currentStats.firstRecovered, currentStats.halfRecovered,
         currentStats.lastRecovered);
  printf("======== First Dead %d,Half Dead %d, Last Dead %d "
         "========\n",
         currentStats.firstDead, currentStats.halfDead, currentStats.lastDead);

  free(agentsCPU);
  cudaFree(devStates);
  cudaFree(agentsGPU);
  cudaFree(devRadiusMaxMovLocal);
}

/*
    Handles the days simulating the probability of getting the virus
    and the struggles of getting it.
*/

void simulate() {
  vector<Agent> agents;
  clock_t start_cpu;
  clock_t end_cpu;
  float cpu_time_enlapsed = 0;
  int day = 0, movement = 0;
  Stats currentStats = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Stats *currentStatsPointer = &currentStats;

  start_cpu = clock();
  initAgents(agents);
  end_cpu = clock();
  cpu_time_enlapsed += end_cpu - start_cpu;

  while (day < numDays) { // days of the simulation
    movement = 0;

    while (movement < maxNumMovDay) {
      for (auto &agent : agents) {

        // Agents Contagion CPU Implementation
        start_cpu = clock();
        contagion(&agent, agents);
        end_cpu = clock();
        cpu_time_enlapsed += end_cpu - start_cpu;

        // Agents Movility CPU Implementation
        start_cpu = clock();
        movility(&agent);
        end_cpu = clock();
        cpu_time_enlapsed += end_cpu - start_cpu;
      }
      movement++;
    }
    for (auto &agent : agents) {

      // Agents Extern Contagion CPU Implementation
      start_cpu = clock();
      externContagion(&agent);
      end_cpu = clock();
      cpu_time_enlapsed += end_cpu - start_cpu;

      // Agents Inffection Effects CPU Implementation
      start_cpu = clock();
      contagionEffects(&agent);
      end_cpu = clock();
      cpu_time_enlapsed += end_cpu - start_cpu;

      // Agents Decease CPU Implementation
      start_cpu = clock();
      decease(&agent);
      end_cpu = clock();
      cpu_time_enlapsed += end_cpu - start_cpu;
    }
    getStats_CPU(agents, currentStatsPointer, day);
    printf("Day: %d, New infected: %d, New  Recovered: %d, New "
           "Dead: %d.\n",
           day, currentStats.newContagion, currentStats.newRecovered,
           currentStats.newDeads);
    if (day % 5 == 0 && day != 0) {
      printf("===== Day: %d, Accumaled infected: %d, Accumaled  Recovered:%d,"
             "Accumaled "
             "Dead: %d. =====\n",
             day, currentStats.accumulateContagion,
             currentStats.accumulateRecovered, currentStats.accumulateDeads);
    }
    day++;
  }

  printf("Time Enlapsed in CPU Implementation: %f ms.\n",
         (cpu_time_enlapsed / CLOCKS_PER_SEC) * 1000);
  getStats_CPU(agents, &currentStats, day);
  printf("======== First Contagion %d, Half Contagion %d, Last Contagion %d "
         "========\n",
         currentStats.firstContagion, currentStats.halfContagion,
         currentStats.lastContagion);
  printf("======== First Recovered %d,Half Recovered %d, Last Recovered %d "
         "========\n",
         currentStats.firstRecovered, currentStats.halfRecovered,
         currentStats.lastRecovered);
  printf("======== First Dead %d,Half Dead %d, Last Dead %d "
         "========\n",
         currentStats.firstDead, currentStats.halfDead, currentStats.lastDead);
}

// /*
//     Function to check if there's someone surrounding (when the agent is not
//     infected) and check if they may get infected by some of the agents near
//     to the actual agent and if they currently are infected.
// */

void contagion(Agent *agent, vector<Agent> &agents) {

  // Check if isn't infected
  if (agent->status != 0) {
    return;
  }
  for (auto &agent2 : agents) {

    float newState = randomFloat(0.0, 1.0);
    // Check if neighbors to a distance of 1 meter
    int itGetInfected = 0;

    if (agent2.status > 0) {
      float distance = EuclideanDistance_CPU(*agent, agent2);
      if (distance <= 1.0) {

        itGetInfected = 1;
        if (newState <= agent->contagionProba && itGetInfected == 1) {
          agent->status = 1;
          break;
        }
      }
    }
  }
}

/*
 In a place, people doesn't stay at the same place for the whole time.
 The may move along the area (short or long run).
*/

void movility(Agent *agent) {
  int actualX = agent->posX;
  int actualY = agent->posY;

  float itsMoving = randomFloat(0.0, 1.0);

  if (agent->status < 0) {
    return;
  }
  if (itsMoving <= agent->movProba) {
    int newX = actualX;
    int newY = actualY;

    float nearMovement = randomFloat(0.0, 1.0);
    int movX = 0, movY = 0;
    int validMovement = 1;
    // Moving near
    if (nearMovement <= agent->shortDistanceMovProba) {
      do {
        validMovement = 1;
        movX = int(2 * (randomFloat(0.0, 1.0)));
        movY = int(2 * (randomFloat(0.0, 1.0)));

        if (movX > radiusMaxMovLocal) {
          movX = radiusMaxMovLocal;
        }
        if (movY > radiusMaxMovLocal) {
          movY = radiusMaxMovLocal;
        }

        if (actualX + movX >= xSize || actualX + movX < 0 ||
            actualY + movY >= ySize || actualY + movY < 0) {
          validMovement = 0;
          movX = 0, movY = 0;
        }

      } while (validMovement == 0);

    } else { // Move long distance
      do {
        validMovement = 1;
        movX = xSize * int(randomFloat(0.0, 1.0));
        movY = ySize * int(randomFloat(0.0, 1.0));

        if (actualX + movX >= xSize || actualX + movX < 0 ||
            actualY + movY >= ySize || actualY + movY < 0) {
          validMovement = 0;
          movX = 0, movY = 0;
        }
      } while (validMovement == 0);
    }
    newX += movX;
    newY += movY;

    agent->posX = newX;
    agent->posY = newY;
  }
}

/*
 The probability of being infected somewhere else
 the place the agents most frequent (outside the
 space we are simulating).
*/

void externContagion(Agent *agent) {
  int status = agent->status;
  if (status == 0) {
    if (randomFloat(0.0, 1.0) <= agent->extContagionProba) {
      agent->status = 1;
    }
    if (agent->status == 1) {
      contagions++;
    }
  }
}

/* This function checks the situation of all infected agents,
 *  simulating the days fighting the disease and the days before
 * the symptoms start to be present in the agent.
 */
void contagionEffects(Agent *agent) {
  if (agent->status > 0) {
    if (agent->incubationTime == 0) {
      agent->status = -1;
    }
    agent->incubationTime -= 1;

  } else if (agent->status < 0 && agent->status != -2) {
    agent->recoveryTime--;
  }
}

/* Function that handles the probability of dying from
 * the disease (people alreadly infected).
 */
void decease(Agent *agent) {
  if (agent->status == -1 && agent->recoveryTime > 0) {
    if (randomFloat(0.0, 1.0) <= agent->deathProba) {
      // status == 2 ---> DEATH
      agent->status = -2;
      deaths++;
    }
  }
}

__host__ int randomInt(int limiteInferior, int limiteSuperior) {
  int randomNum;
  /* generate  number between limiteInferior and limiteSuperior: */
  randomNum = rand() % limiteSuperior + limiteInferior;
  return randomNum;
}

__host__ float randomFloat(float a, float b) {
  float random = ((float)rand()) / (float)RAND_MAX;
  float diff = b - a;
  float r = random * diff;
  return a + r;
}