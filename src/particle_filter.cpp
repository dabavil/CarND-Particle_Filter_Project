/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	//Lots of values for this parameter work, even down to ~20, but the accuracy suffers. Above 500 gets noticeably slow
	num_particles = 50; // Value of 50 sacrifices average ~1cm of precision on X axis vs value of 500

	default_random_engine gen; //random number to sample from our gauss distr.

	//define the gaussian distributions around initial GPS estimates
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_yaw(theta, std[2]);

	for(int i = 0; i < num_particles; i++)
	{
		Particle p; // instantiate a single particle

		//set x, y, and yaw as random sample from the gaussian distribution
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_yaw(gen);
		//set initial weight to 1
		p.weight = 1.0;
		//set particle id
		p.id = i;

		//now add particle and its weight to the respective vector
		particles.push_back(p);
		weights.push_back(1.0);

	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) 
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	//Make some noise! AKA Create gaussians to be used around position estimate
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_yaw(0, std_pos[2]);

	//**********************************
	//******Apply current controls to the latest state estimate to predict state at t+1
	
	//Iterate over each particle
	for(int i = 0; i < num_particles; i++)
	{
		//capture values of current state for better readibility
		double x_t = particles[i].x;
		double y_t = particles[i].y;
		double yaw_t = particles[i].theta;

		//predict x and y; depends on whether yaw rate is 0 or not
		if(fabs(yaw_rate) < 0.0001)
		{ 	// straight line motion model
			particles[i].x = x_t + velocity * sin(yaw_t) * delta_t;
			particles[i].y = y_t + velocity * cos(yaw_t) * delta_t;
			//yaw does not change in straight line motion
		} else
		{
			//predict yaw at t+1
			double yaw_t1 = yaw_t + yaw_rate * delta_t;
			particles[i].x = x_t + velocity / yaw_rate * (sin(yaw_t1) - sin(yaw_t));
			particles[i].y = y_t + velocity / yaw_rate * (cos(yaw_t) - cos(yaw_t1));
			particles[i].theta = yaw_t1;
		}

		//add some noise
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_yaw(gen);
	} // cout<<"Ended prediction step for a single particle."
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) 
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


	// Clear the old weights; Below we will set new weights depending on the likelihood of the related particle to be correct position
	weights.clear();

	// Set some constants to improve readibility and minimize recals
	const double c = 2 * M_PI * std_landmark[0] * std_landmark[1];

	//Iterate over each particle 
	for(int i = 0; i < num_particles; i++)
	{
		
		//Capture x,p, and yaw values of the particle for better readability
		double x_p = particles[i].x;
    	double y_p = particles[i].y;
    	double yaw_p = particles[i].theta;

    	std::vector<LandmarkObs> predicted;
    
    	//Clear any current particle associations and sense values
    	particles[i].associations.clear();
    	particles[i].sense_x.clear();
    	particles[i].sense_y.clear();
    	double weight = 1;
    	
    	//Iterate over each observation
    	for(int j = 0; j < observations.size(); j++)
    	{
    		//Extract the observation x and y data, in car coordinates
    		double o_x = observations[j].x;
      		double o_y = observations[j].y;

      		//Transform the observation data into map coordinates
      		double o_map_x = o_x * cos(yaw_p) - o_y * sin(yaw_p) + x_p;
      		double o_map_y = o_x * sin(yaw_p) + o_y * cos(yaw_p) + y_p;
      	
      		//Test if not outside of sensor range, if so return to start of the loop
      		if(dist(o_map_x, o_map_y, x_p, y_p) > sensor_range)
      		{
      			continue;
      		}
      	
      		particles[i].sense_x.push_back(o_map_x);
    		particles[i].sense_y.push_back(o_map_y);
    		
    		double min_distance = 10000000; // large init number
     		int min_k=-1; // init value for k

     		//Iterate over list of landmarks, calculate distance to observation
    		for(int k = 0; k < map_landmarks.landmark_list.size(); k++)
    		{
        		double land_x = map_landmarks.landmark_list[k].x_f;
        		double land_y = map_landmarks.landmark_list[k].y_f;       
        		
        		double distance = dist(land_x, land_y, o_map_x, o_map_y);

        		
        		// See if this distance to landmark is smaller that the smallest previous found, if so, store the current one	
        		if(distance < min_distance)
        		{
          			min_distance = distance;
          			min_k = k;
        		}
      		}

      		double land_x = map_landmarks.landmark_list[min_k].x_f;
      		double land_y = map_landmarks.landmark_list[min_k].y_f;


      		//Populate the particle vector w. new associations
      		particles[i].associations.push_back(map_landmarks.landmark_list[min_k].id_i);


      		//Calculate the weight using a mult-variate Gaussian distribution
      		weight = weight * exp(-0.5 * (pow((land_x - o_map_x) / std_landmark[0],2) + pow((land_y - o_map_y) / std_landmark[1],2))) / c;
    	} 


    	particles[i].weight=weight;
    	weights.push_back(weight); 
  	}
 
}

void ParticleFilter::resample() 
{
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	//		***From docs:***
	//			std::discrete_distribution produces random integers on the interval [0, n), 
	// 			where the probability of each individual integer i is defined as wi/S,
	// 			that is the weight of the ith integer divided by the sum of all n weights.


	//Helper to generate random nums
	default_random_engine gen;

	//Create a discrete distro for particles based on latest weights
 	discrete_distribution<int> particle_distr(weights.begin(), weights.end());
 	
 	//Clear old weights vector- will be replaces by the corresponding weights of the new particles
 	weights.clear();

 	//Create vector to hold the resampled particles
 	
 	std::vector<Particle> particles_temp;

  
  	//Iterate over the discreddistribution, pick new particle, and add it to resampled particle vector
 	for(int i=0; i < num_particles; i++)
 	{
 		//Pick index of a particle from the distribution. The higher was weight of this particle, the higher probability of picking it.
 		int selected_i = particle_distr(gen);
 		//Include the particle with this index in the temporary particle vector
 		particles_temp.push_back(particles[selected_i]);
 		//Include the weight associated w. this particle in the weights vector
 		weights.push_back(particles[selected_i].weight);
  	}
  	
  	//Update the particles vector w. the newly resampled particles
	particles = particles_temp;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear any previous associations and sense_x,y vectors
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
