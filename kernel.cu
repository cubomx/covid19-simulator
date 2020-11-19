#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>  /* time */
#include <vector>
#include <algorithm>

using namespace std;

int randomInt(int, int);
float randomFloat(float, float);

class Agent {
public:

    float contagionProba; // 0.02 - 0.03
    float extContagionProba; // 0.02 - 0.03
    float deathProba; // 0.007 - 0.07
    float movProba; // 0.3 - 0.5
    float shortDistanceMovProba; // 0.7 - 0.9
    int incubationTime; // 5 - 6
    int recoveryTime; // 14

    int status; /* (0) No infected, (1) infected, (-1) Quarantine,
    (-2) Decease
    */
    int posX; // 0 - p
    int posY; // 0 - q
    void generate(int x, int y) {
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

//Quantity of agents
const int numAgents = 10240;
//Days of duration of the simulation
const int numDays = 9;
//Maximum number of movements per day
const int maxNumMovDay = 10;
//Maximum radius of local movement
const int radiusMaxMovLocal = 5;
//Meters of distance that the virus can travel
const int distanceContagion = 1;

int xSize = 500, ySize = 500;

int deaths = 0;

int contagions = 0;

void initAgents(vector <Agent>& agents, Agent* map[][500]) {
    for (int i = 0; i < numAgents; i++) {
        int posX = 0;
        int posY = 0;

        do {
            posX = randomInt(0, 500);
            posY = randomInt(0, 500);
        } while (map[posX][posY] != NULL);

        Agent newAgente = Agent();

        newAgente.generate(posX, posY);

        agents.push_back(newAgente);
        map[posX][posY] = &newAgente;
    }
}

void simulate(vector <Agent>&, Agent* [][500]);
void contagion(Agent*, Agent* [][500]);
void movility(Agent*, Agent* [][500]);
void externContagion(Agent*);
void contagionEffects(Agent*);
void decease(Agent*);


int main() {
    srand(time(NULL));

    static Agent* map[500][500] = { 0 };
    vector<Agent> agents;

    initAgents(agents, map);
    simulate(agents, map);

    cout << "Total sum of deaths in " << numDays << " days is: " << deaths << "\n";
    cout << "Total sum of contagions in " << numDays << " days is: " << contagions << "\n";
}

/*
    Handles the days simulating the probability of getting the virus
    and the struggles of getting it.
*/

void simulate(vector <Agent>& agents, Agent* map[][500]) {
    int day = 0, movement = 0;
    printf("%lu\n", agents.size());
    while (day < numDays) { // days of the simulation
        movement = 0;

        while (movement < maxNumMovDay) {
            for (auto& agent : agents) {
                contagion(&agent, map);
                movility(&agent, map);
            }
            movement++;
        }
        for (auto& agent : agents) {
            externContagion(&agent);
            contagionEffects(&agent);
            decease(&agent);
        }
        day++;
    }
}

/*
    Function to check if there's someone surrounding (when the agent is not
    infected) and check if they may get infected by some of the agents near
    to the actual agent and if they currently are infected.
*/


void contagion(Agent* agent, Agent* map[][500]) {
    // Check if isn't infected
    if (agent->status == 0) {
        int x = agent->posX;
        int y = agent->posY;
        int newState = randomFloat(0.0, 1.0);
        // Check if neighbors to a distance of 1 meter
        int itGetInfected = 0;
        if (x + 1 < xSize) {
            if (map[x + 1][y] != nullptr && map[x + 1][y]->status > 0) {
                itGetInfected = 1;
            }
        }
        if (x - 1 >= 0) {
            if (map[x - 1][y] != nullptr && map[x - 1][y]->status > 0) {
                itGetInfected = 1;
            }
        }

        if (map[x][y + 1] != nullptr && y + 1 < ySize) {
            if (map[x][y + 1]->status > 0) {
                itGetInfected = 1;
            }
        }

        if (y - 1 >= 0) {
            if (map[x][y - 1] != nullptr && map[x][y - 1]->status > 0) {
                itGetInfected = 1;
            }
        }

        if (newState <= agent->contagionProba && itGetInfected == 1) {
            agent->status = 1;
            contagions++;
        }

    }

}

/*
 In a place, people doesn't stay at the same place for the whole time.
 The may move along the area (short or long run). 
*/

void movility(Agent* agent, Agent* map[][500]) {
    int actualX = agent->posX;
    int actualY = agent->posY;

    int itsMoving = randomFloat(0.0, 1.0);

    if (itsMoving <= agent->movProba) {
        int newX = actualX;
        int newY = actualY;

        int nearMovement = randomFloat(0.0, 1.0);
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

                if (actualX + movX >= xSize || actualX + movX < 0 || actualY + movY >= ySize || actualY + movY < 0) {
                    validMovement = 0;
                    movX = 0, movY = 0;
                }

            } while (map[actualX + movX][actualY + movY] != NULL && validMovement == 0);


        }
        else { // Move long distance
            do {
                validMovement = 1;
                movX = xSize * (randomFloat(0.0, 1.0));
                movY = ySize * (randomFloat(0.0, 1.0));

                if (actualX + movX >= xSize || actualX + movX < 0 || actualY + movY >= ySize || actualY + movY < 0) {
                    validMovement = 0;
                    movX = 0, movY = 0;
                }
            } while (map[actualX + movX][actualY + movY] != NULL && validMovement == 0);
        }
        newX += movX;
        newY += movY;
        map[actualX][actualY] = nullptr;

        agent->posX = newX;
        agent->posY = newY;

        map[newX][newY] = agent;

    }
}

/*
 The probability of being infected somewhere else
 the place the agents most frequent (outside the 
 space we are simulating).
*/

void externContagion(Agent* agent) {
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
void contagionEffects(Agent* agent) {
    if (agent->status > 0) {
        if (agent->incubationTime == 0) {
            agent->status = -1;
        }
        agent->incubationTime -= 1;

    }
    else if (agent->status < 0) {
        agent->recoveryTime--;
    }
}

/* Function that handles the probability of dying from
* the disease (people alreadly infected).
*/
void decease(Agent* agent) {
    if (agent->status == -1 && agent->recoveryTime > 0) {
        if (randomFloat(0.0, 1.0) <= agent->deathProba) {
            // status == 2 ---> DEATH
            agent->status = -2;
            deaths++;
        }
    }

}

int randomInt(int limiteInferior, int limiteSuperior) {
    int randomNum;
    /* generate  number between limiteInferior and limiteSuperior: */
    randomNum = rand() % limiteSuperior + limiteInferior;
    return randomNum;
}

float randomFloat(float a, float b) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
    
}