//to compile - g++-13 -std=c++23 main.cpp 
//to compile - g++-13 -std=c++23 -O1 -g -Wall -Wextra -Wshadow -fsanitize=address,undefined main.cpp
//to run - ./a.out

#include <iostream>
#include <array>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>

bool are_equal(float a, float b, float epsilon=0.00001f) {
	return std::abs(a - b) < epsilon;
}

// an MNIST sample holding the image (28*28 float values) and a label (0 or 1)
struct Sample {
	std::vector<float> image;
	float target{};
};
using Dataset = std::vector<Sample>;


// load the MNIST dataset from a text file
Dataset load_dataset(const std::string& filepath) {
	Dataset dataset;

	std::ifstream file(filepath);
	if (!file) {
		std::cout << "Could not read dataset!" << std::endl;
		return dataset;
	}
	std::string tmp;
	char c;
	bool is_label = true;
	while (file.get(c)) {
		tmp += c;
		if (c == ',' || c == ';') {
			int val = std::stoi(tmp);
			tmp.clear();
			if (is_label) {
				dataset.push_back(Sample());
				dataset.back().target = static_cast<float>(val);
				is_label = false;
			}
			else {
				dataset.back().image.push_back(val / 255.0f);
			}
			if (c == ';') {
				is_label = true;
			}
		}
	}
	return dataset;
}

int layers = 3;
const size_t BATCH_SIZE = 32;
std::vector<int> num_neurons = {784, 10, 1};
std::vector<std::vector<std::vector<float>>> weights = {};
std::vector<std::vector<float>> biases = {};


float sigmoid(float x) {
    return 1 / (1 + std::exp(-x));
}

float derivative_sigmoid(float x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

std::vector<float> forward(std::vector<float> input,
	std::vector<std::vector<float>> &neuron_activations,
	std::vector<std::vector<float>> &neuron_pre_activations) {

	neuron_activations.push_back(input);

    for(int i = 0; i < std::size(weights); i++) {
        std::vector<float> temp(num_neurons[i + 1], 0);
        for (int j = 0; j < std::size(weights[i]); j++) {
            for (int k = 0; k < std::size(weights[i][j]); k++) {
                temp[k] += input[j] * weights[i][j][k];
            }
        }
        input = temp;
		for(int x = 0; x < std::size(input); x++) {
            input[x] = input[x] + biases[i][x];
        }
		neuron_pre_activations.push_back(input);
        for(int x = 0; x < std::size(input); x++) {
            input[x] = sigmoid(input[x]);
        }
		neuron_activations.push_back(input);
    }

    return input;
}

float loss_function(Sample &sample,
						std::vector<std::vector<float>> &neuron_activations,
						std::vector<std::vector<float>> &neuron_pre_activations) {
	auto estimate = forward(sample.image, neuron_activations, neuron_pre_activations);
	float estimate_value = estimate[0];
	float target = sample.target;
	float loss = -(std::log(estimate_value)*target + std::log(1 - estimate_value)*(1 - target));
	return loss;
}

float loss_batched(Dataset &batch,
					std::vector<std::vector<std::vector<float>>> &neuron_activations_all,
					std::vector<std::vector<std::vector<float>>> &neuron_pre_activations_all) {
	float average_loss = 0;
	for(size_t i = 0; i < std::size(batch); i++) {
		std::vector<std::vector<float>> neuron_activations;
		std::vector<std::vector<float>> neuron_pre_activations;
		average_loss += loss_function(batch[i], neuron_activations, neuron_pre_activations)
		 / static_cast<float>(std::size(batch));
		neuron_activations_all.push_back(neuron_activations);
		neuron_pre_activations_all.push_back(neuron_pre_activations);
	}
	return average_loss;
}

float accuracy(Dataset &batch) {
	return 0;
}

void sgd_update(Dataset &batch) {
	
	// get the losses
	std::vector<std::vector<std::vector<float>>> neuron_activations_all;
	std::vector<std::vector<std::vector<float>>> neuron_pre_activations_all;

	float loss = loss_batched(batch, neuron_activations_all, neuron_pre_activations_all);
	// std::cout << loss << std::endl;

	// compute the derivaties
	std::vector<std::vector<std::vector<float>>> weight_gradients;

	// 1. Size the number of layers
	weight_gradients.resize(weights.size());

	for (size_t i = 0; i < weights.size(); ++i) {
		// 2. Size the number of input neurons for this layer
		weight_gradients[i].resize(weights[i].size());
		
		for (size_t j = 0; j < weights[i].size(); ++j) {
			// 3. Size the number of output neurons for this connection
			// And initialize values to 0.0f
			weight_gradients[i][j].resize(weights[i][j].size(), 0.0f);
		}
	}

	std::vector<std::vector<std::vector<float>>> deltas;
	deltas.resize(neuron_activations_all.size());

	for (size_t i = 0; i < neuron_activations_all.size(); ++i) {
		deltas[i].resize(neuron_activations_all[i].size());
		for (size_t j = 0; j < neuron_activations_all[i].size(); ++j) {
			deltas[i][j].resize(neuron_activations_all[i][j].size(), 0.0f);
		}
	}

	for (size_t b = 0; b < std::size(batch); b++) {
		for (int l = std::size(neuron_activations_all[b]) - 1; l > 0; l--) {
			for (size_t j = 0; j < std::size(neuron_activations_all[b][l]); j++) {
                if (l == std::size(neuron_activations_all[b]) - 1) {
                    float loss_derivative_pre_activation = -batch[b].target + neuron_activations_all[b][l][j];
                    deltas[b][l][j] = loss_derivative_pre_activation;
                }
                else {
                    for (int k = 0; k < std::size(neuron_activations_all[b][l + 1]); k++) {
                        deltas[b][l][j] += deltas[b][l + 1][k] * weights[l][j][k];
                    }
                    deltas[b][l][j] *=  derivative_sigmoid(neuron_pre_activations_all[b][l - 1][j]);
                }
			}
		}
	}

    float lr = 0.01;

	// update the weights
    for (size_t b = 0; b < std::size(batch); b++) {
        for (int l = 0; l < std::size(weights); l++) {
            for (int i = 0; i < std::size(weights[l]); i++) {
                for (int j = 0; j < std::size(weights[l][i]); j++) {
                    weights[l][i][j] -= lr*deltas[b][l + 1][j]*neuron_activations_all[b][l][i] 
						/ static_cast<float>(std::size(batch));
                }
            }
        }
    }

    // update the biases
    for (size_t b = 0; b < std::size(batch); b++) {
        for (int l = 0; l < std::size(biases); l++) {
            for (int i = 0; i < std::size(biases[l]); i++) {
                biases[l][i] -= lr*deltas[b][l + 1][i] / static_cast<float>(std::size(batch));
            }
        }
    }
}


float calculate_accuracy(Dataset &dataset) {
	int correct_predictions = 0;
	for (size_t i = 0; i < std::size(dataset); i++) {
		std::vector<std::vector<float>> neuron_activations;
		std::vector<std::vector<float>> neuron_pre_activations;
		auto output = forward(dataset[i].image, neuron_activations, neuron_pre_activations);
		float predicted_label = output[0] >= 0.5f ? 1.0f : 0.0f;
		if (are_equal(predicted_label, dataset[i].target)) {
			correct_predictions++;
		}
	}
	return static_cast<float>(correct_predictions) / static_cast<float>(std::size(dataset));
}

int main() {
    auto dataset = load_dataset("data/mnist_train.txt");
    std::cout << "dataset loaded!" << std::endl;

	for (size_t s = 0; s < dataset.size(); ++s) {
		if (dataset[s].image.size() != 784) {
			std::cout << "Bad sample " << s << " image.size()=" << dataset[s].image.size() << "\n";
			return -1;
		}
	}

    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::normal_distribution<float> d(0.0f, 1.0f); // Mean 0, StdDev 1

    for(int i = 0; i < std::size(num_neurons) - 1; i++) {
        std::vector<std::vector<float>> layer_weights(num_neurons[i], std::vector<float>(num_neurons[i + 1]));

        for (int j = 0; j < num_neurons[i]; ++j) {
            for (int k = 0; k < num_neurons[i + 1]; ++k) {
                layer_weights[j][k] = d(gen);
            }
        }
        weights.push_back(layer_weights);
    }

	for(int i = 1; i < std::size(num_neurons); i++) {
		std::vector<float> layer_biases(num_neurons[i]);
		for(int j = 0; j < std::size(layer_biases); j++) {
			layer_biases[j] = d(gen);
		}
		biases.push_back(layer_biases);
	}

	auto test_dataset = load_dataset("data/mnist_test.txt");
	std::cout << "test dataset loaded!" << std::endl;

	float acc = calculate_accuracy(test_dataset);
	std::cout << "Test Accuracy: " << acc * 100.0f << "%" << std::endl;

	for(int epoch = 0; epoch < 50; epoch++) {
		for(int i = 0; i < static_cast<int>(std::size(dataset) / BATCH_SIZE) - 1; i++) {
			size_t start_idx = i * BATCH_SIZE;
			Dataset batch(dataset.begin() + start_idx, dataset.begin() + start_idx + BATCH_SIZE);
			sgd_update(batch);
		}
		std::cout << "Epoch " << epoch + 1 << " completed." << std::endl;
		acc = calculate_accuracy(dataset);
		std::cout << "Training Accuracy: " << acc * 100.0f << "%" << std::endl;

		acc = calculate_accuracy(test_dataset);
		std::cout << "Test Accuracy: " << acc * 100.0f << "%" << std::endl;
	}

    return 0;
}