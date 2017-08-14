/* particle_filter.cpp
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang */
#include <random>
//#include <map>
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
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of x, y, theta and their uncertainties from GPS) and all weights to 1. Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 10; // Pick that works.
	
	default_random_engine gen;

	normal_distribution<double> N_x(x, std[0]);
	normal_distribution<double> N_y(y, std[1]);
	normal_distribution<double> N_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = N_x(gen);
		particle.y = N_y(gen);
		particle.theta = N_theta(gen);
		particle.weight = 1;
		particles.push_back(particle);
		weights.push_back(1);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful. http://en.cppreference.com/w/cpp/numeric/random/normal_distribution & http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	for (int i = 0; i < num_particles; i++) {
		double new_x;
		double new_y;
		double new_theta;

		if(fabs(yaw_rate) < 0.0001) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta); // changed from new_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
			new_theta = particles[i].theta;
		} else {
			particles[i].x += velocity/yaw_rate * (sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
			particles[i].y += velocity/yaw_rate * (cos(particles[i].theta)-cos(particles[i].theta + yaw_rate*delta_t));
			new_theta = particles[i].theta + yaw_rate * delta_t;
		}
		normal_distribution<double> N_x(particles[i].x, std_pos[0]);
		normal_distribution<double> N_y(particles[i].y, std_pos[1]);
		normal_distribution<double> N_theta(new_theta, std_pos[2]);

		particles[i].x = N_x(gen);
		particles[i].y = N_y(gen);
		particles[i].theta = N_theta(gen);
	}
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to implement this method and use it as a helper during the updateWeights phase.
	
	// https://discussions.udacity.com/t/c-help-with-dataassociation-method/291220
	// https://discussions.udacity.com/t/implementing-data-association/243745/7
	for (int i = 0; i < observations.size(); i++) {
		double min_distance = numeric_limits<double>::max();;
		int ob_id = -1;
		for (int j = 0; j < predicted.size(); j++) {
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			int pr_id = predicted[j].id;
			if (distance < min_distance) {
				min_distance = distance;
				ob_id = pr_id;
			}
		}
		observations[i].id = ob_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located according to the MAP'S coordinate system. You will need to transform between the two systems. Keep in mind that this transformation requires both rotation AND translation (but no scaling). The following is a good resource for the theory: https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm and the following is a good resource for the actual equation  to implement (look at equation 3.33 http://planning.cs.uiuc.edu/node99.html

	weights.clear();

	for(int p = 0; p < num_particles; p++) {
		vector<LandmarkObs> trans_observations;
		vector<LandmarkObs> predicted;
		double weight = 1.0;
		LandmarkObs obs;
		// Observations of map Coordinates
		for(int i = 0; i < observations.size(); i++) {
			LandmarkObs trans_obs;
			obs = observations[i];
			trans_obs.x = (obs.x*cos(particles[p].theta))-(obs.y*(sin(particles[p].theta)))+particles[p].x;
			trans_obs.y = (obs.x*sin(particles[p].theta))+(obs.y*(cos(particles[p].theta)))+particles[p].y;
			trans_observations.push_back(trans_obs);	
		}
		particles[p].weight = 1.0;
		for(int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			double closet_dis = sensor_range;	
      double distance = dist(particles[p].x, particles[p].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
      if (distance < closet_dis) {
      	LandmarkObs landmark;
      	landmark.x = map_landmarks.landmark_list[j].x_f;
      	landmark.y = map_landmarks.landmark_list[j].y_f;
      	landmark.id = map_landmarks.landmark_list[j].id_i;
      	predicted.push_back(landmark);
      }
    }
    // Sending predicted and observations
    dataAssociation(predicted, trans_observations);
    
    for(int k = 0; k < trans_observations.size(); k++) {
			// Calculating for the Multivariate-Gaussian Probability
   		// https://discussions.udacity.com/t/output-always-zero/260432/32
			double meas_x = trans_observations[k].x;
			double meas_y = trans_observations[k].y;
			double mu_x = 1.0;
			double mu_y = 1.0;
			double mult_weight = 1.0;
			for (int n = 0; n < predicted.size(); n++) {
				if(predicted[n].id == trans_observations[k].id) {
					mu_x = predicted[n].x;
					mu_y = predicted[n].y;
					//cout<<"mu_x: "<<mu_x<<endl;
					//cout<<"mu_y: "<<mu_y<<endl;
				}
			}
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			double muy = meas_y - mu_y;
			double my = muy * muy;
			double mux = meas_x - mu_x;
			double mx = mux * mux;
			double stdx = 2 * pow(std_x, 2);
			double stdy = 2 * pow(std_y, 2);
			double gauss_norm = exp(-1*(mx/(stdx) + (my/(stdy))))/(2 * M_PI * std_x * std_y);

			//cout<<"gauss_norm: "<<gauss_norm<<endl;			
			if (gauss_norm > 0) {
				weight *= gauss_norm;
				//cout<<"weight: "<<weight<<endl;
			}
		}
		weights.push_back(weight);
		particles[p].weight = weight;
		//cout<<"weight: "<<weight<<endl;
		//cout<<"particles: "<<particles[p].weight<<endl;
	}	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here. http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());

	vector<Particle> resample_particles;
	for(int i = 0; i < num_particles; i++) {
		resample_particles.push_back(particles[distribution(gen)]);
	}
	particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, vector<int> associations, vector<double> sense_x, std::vector<double> sense_y) {
	// Particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to associations: The landmark id that goes along with each listed association
	// sense_x:  the associations x mapping already converted to world coordinates
	// sense_y:  the associations y mapping already converted to world coordinates

	// Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // Get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // Get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // Get rid of the trailing space
  return s;
}