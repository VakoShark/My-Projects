// https://micro-os-plus.github.io/develop/sutter-101/
/* This link contains exactly 100 different best coding practices
   for C++. Obviously not all of them apply to a simple program
   like this one, but I still tried to implement the practices
   that were relevant when applicable. */

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <random>

using namespace std;

constexpr auto DBL_EPS = 1e-12;

/* Functions declared */
vector<vector<double>> readData(char* file_name, int observations, int dimensions);
void normalize(vector<vector<double>>& data);
double randSelection(char* file_name, int clusters, vector<vector<double>>& centers_selection,
	vector<vector<double>>& data, int observations, int dimensions);
double randPartition(char* file_name, int clusters, vector<vector<double>>& centers_partition,
	vector<vector<double>>& data, int observations, int dimensions);
double k_means(char* file_name, const int clusters,
	vector<vector<double>>& data, vector<vector<double>>& centers, int observations, int dimensions,
	vector<double>& n_vals, vector<int>& cluster_idx);
double calinskiHarabasz(double sse, int clusters, int observations, int dimensions, vector<vector<double>>& data,
	vector<vector<double>>& centers, vector<double> n);
double silhouetteWidth(int observations, int dimensions, vector<int>& cluster_idx, vector<vector<double>>& data,
	vector<vector<double>>& centers, int clusters, vector<double> n);

int main(int argc, char* argv[])
{
	char* file_name;
	int clusters;
	int max_iters;
	double conv_thresh;
	int runs;
	int observations = 0;
	int dimensions = 0;
	vector<vector<double>> data;
	vector<vector<double>> centers;
	vector<double> SSE;
	double sse = 0;
	double min_sse;
	double ch_value = 0;
	double sw_value = 0;
	int k_min = 2;
	int k_max;
	vector<double> n;
	vector<double> n_vals;
	vector<int> cluster_idx;
	vector<int> cluster_idx_capture;

	/* Data validation checks */
	if (argc == 6)
	{
		/* Data file name */
		file_name = argv[1];

		/* Number of clusters (integer greater than 1) */
		if (atoi(argv[2]) <= 1)
		{
			printf("Invalid number of clusters supplied (must be larger than 1).\n");
			return 0;
		}
		clusters = atoi(argv[2]);

		/* Number of maximum iterations (positive integer) */
		if (atoi(argv[3]) < 1)
		{
			printf("Invalid number of iterations supplied (must be positive).\n");
			return 0;
		}
		max_iters = atoi(argv[3]);

		/* Convergence threshold (positive real number) */
		if (atof(argv[4]) <= 0)
		{
			printf("Invalid convergence threshold supplied (must be positive).\n");
			return 0;
		}
		conv_thresh = atof(argv[4]);

		/* Number of runs (positive integer) */
		if (atoi(argv[5]) < 1)
		{
			printf("Invalid number of runs supplied (must be positive).\n");
			return 0;
		}
		runs = atoi(argv[5]);
	}
	else
	{
		printf("Invalid number of arguments supplied, must be file name, number of clusters, maximum iterations, convergance threshold, and runs.\n");
		return 0;
	}

	ifstream in_file(file_name);
	if (in_file.is_open())
	{
		/* Getting the number of observations and the dimensionality */
		in_file >> observations;
		in_file >> dimensions;
	}
	else
	{
		cerr << "Error opening file (in_file)\n" << endl;
	}
	in_file.close();

	string new_file = string(file_name) + "_new.txt";
	ofstream out_file(new_file);

	std::cout << "Reading the data..." << endl;
	/* Reading the data into a 2d vector */
	data = readData(file_name, observations, dimensions);

	/* Normalizing the data using min-max normalization */
	normalize(data);

	k_max = int(round(sqrt(observations / 2)));

	std::cout << "Generating random centers and running k-means..." << endl;
	if (out_file.is_open())
	{
		/* Running the algorithms multiple times to find the optimal CH and SW values*/
		for (int k = k_min; k < k_max; k++)
		{
			std::cout << "K: " << k << endl;
			min_sse = 10000.0;
			/* Running k-means for the random partition initialization method */
			for (int i = 0; i < runs; i++)
			{
				/* Calling the random center selection function */
				sse = randPartition(file_name, k, centers, data, observations, dimensions);
				
				SSE.push_back(min_sse);
				int j = 1;
				do
				{
					/* Running the k-means algorithm on the data and centers from randPartition and
					   sending the output to new text files */
					sse = k_means(file_name, k, data, centers, observations, dimensions, n_vals, cluster_idx);
					SSE.push_back(sse);
					j++;
				} while (j < max_iters + 1 && (conv_thresh < (SSE[j - 2] - SSE[j - 1]) / SSE[j - 2]));

				if (SSE.back() < min_sse)
				{
					min_sse = SSE.back();
					n = n_vals;
					cluster_idx_capture = cluster_idx;
				}
				SSE.clear();
			}
			/* Calculating the Calinski-Harabasz index value and Silhouette Width index value */
			ch_value = calinskiHarabasz(min_sse, k, observations, dimensions, data, centers, n);
			sw_value = silhouetteWidth(observations, dimensions, cluster_idx_capture, data, centers, k, n);

			out_file << "k = " << k << endl;
			out_file << ch_value << endl;
			out_file << sw_value << endl << endl;
		}
	}
	else
	{
		cerr << "Error opening file (out_file)\n" << endl;
	}
	out_file.close();

	std::cout << "Done!" << endl;

	return 0;
}

/* The data reading function */
vector<vector<double>> readData(char* file_name, int observations, int dimensions)
{
	ifstream in_file(file_name);
	double point = 0;
	vector<vector<double>> data;
	vector<double> row;

	/* Reading from the text file specified into a 2d vector */
	if (in_file.is_open())
	{
		for (int i = 0; i < observations; i++)
		{
			for (int j = 0; j < dimensions; j++)
			{
				in_file >> point;
				row.push_back(point);
			}
			data.push_back(row);
			row.clear();
		}
	}
	else
	{
		cerr << "Error opening file (in_file)\n" << endl;
	}
	in_file.close();

	return data;
}

/* The min-max normalization function */
void normalize(vector<vector<double>>& data)
{
	/* This function uses the min-max normalization formula to
	   normalize the data to the range of [0,1] */
	vector<double> column;
	for (int i = 0; i < data[i].size(); i++)
	{
		double min = 1000000.0;
		double max = -1.0;
		/* Iterating over the data by column */
		for (int j = 0; j < data.size(); j++)
		{
			column.push_back(data[j].at(i));
			if (data[j].at(i) < min)
			{
				min = data[j].at(i);
			}
			if (data[j].at(i) > max)
			{
				max = data[j].at(i);
			}
		}

		/* Rewriting the original dataset with the new normalized values */
		for (int j = 0; j < column.size(); j++)
		{
			if (fabs(max - min) < DBL_EPS)
			{
				data[j].at(i) = 0;
			}
			else
			{
				data[j].at(i) = ((data[j].at(i) - min) / (max - min));
			}
		}
		column.clear();
	}
}

/* The random center generating function */
double randSelection(char* file_name, int clusters, vector<vector<double>>& centers_selection, vector<vector<double>>& data, int observations, int dimensions)
{
	centers_selection.clear();

	/* Generating the random positions in the vector for the centers */
	random_device rd;
	mt19937 gen(rd());
	int entries = observations - 1;
	uniform_int_distribution<> distrib(0, entries);
	int position = 0;
	vector<int> centerIdxSelection;
	int dummy = 0;

	/* Selecting the random centers and adding them to centerIdx */
	int i = 0;
	do
	{
		position = distrib(gen);
		if (find(centerIdxSelection.begin(), centerIdxSelection.end(), position) != centerIdxSelection.end())
		{
			/* Do nothing */
			dummy++;
		}
		else
		{
			centerIdxSelection.push_back(position);
			i++;
		}
	} while (i < clusters);

	/* Setting the centers with centerIdxSelection */
	for (int i = 0; i < centerIdxSelection.size(); i++)
	{
		centers_selection.push_back(data.at(centerIdxSelection[i]));
	}

	/* Copying the first part of the k-means algorithm to find the minimun
	   distances for the SSE */
	double sse = 0;
	for (int i = 0; i < observations; i++)
	{

		double min_dist = 10000000.0;
		/* argmin algorithm for distance between points and centers */
		for (int j = 0; j < clusters; j++)
		{
			double difference = 0;
			double new_dist = 0;

			/* Iterating through each of the observations' points */
			for (int k = 0; k < dimensions; k++)
			{
				difference = data[i][k] - centers_selection[j][k];
				new_dist += difference * difference;
			}

			if (new_dist < min_dist)
			{
				min_dist = new_dist;
			}
		}
		sse += min_dist;
	}
	return sse;
}

/* The random partition initialization function */
double randPartition(char* file_name, int clusters, vector<vector<double>>& centers_partition, vector<vector<double>>& data, int observations, int dimensions)
{
	centers_partition.clear();

	/* Creating a 2d cluster values vector */
	 /* Cluster values:
		cluster_vals[i][0] = size of cluster i
		cluster_vals[i][1->dimensions-1] = totals for each dimension of cluster i */
	vector<vector<double>> cluster_vals(clusters, vector<double>(dimensions + 1, 0));
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<> distrib(0, clusters - 1);
	int random = 0;
	vector<double> row;

	/* Randomly assigning each data point to a cluster for the initial centroids */
	for (int i = 0; i < data.size(); i++)
	{
		random = distrib(gen);
		cluster_vals[random][0] += 1;
		for (int j = 1; j < dimensions + 1; j++)
		{
			cluster_vals[random][j] += data[i][j - 1];
		}
	}
	/* Add the centers to a centers vector passed in by reference */
	for (int k = 0; k < clusters; k++)
	{
		for (int l = 1; l < dimensions + 1; l++)
		{
			if (cluster_vals[k][0] == 0)
			{
				row.push_back(0);
			}
			else
			{
				row.push_back(cluster_vals[k][l] / cluster_vals[k][0]);
			}
		}
		centers_partition.push_back(row);
		row.clear();
	}

	/* Copying the first part of the k-means algorithm to find the minimun
	   distances for the SSE */
	double sse = 0;
	for (int i = 0; i < observations; i++)
	{
		double min_dist = 10000000.0;
		/* argmin algorithm for distance between points and centers */
		for (int j = 0; j < clusters; j++)
		{
			double difference = 0;
			double new_dist = 0;

			/* Iterating through each of the observations' points */
			for (int k = 0; k < dimensions; k++)
			{
				difference = data[i][k] - centers_partition[j][k];
				new_dist += difference * difference;
			}

			if (new_dist < min_dist)
			{
				min_dist = new_dist;
			}
		}
		sse += min_dist;
	}

	return sse;
}

/* The k-means algorithm */
double k_means(char* file_name, const int clusters,
	vector<vector<double>>& data, vector<vector<double>>& centers, int observations, int dimensions,
	vector<double>& n_vals, vector<int>& cluster_idx)
{
	n_vals.clear();
	cluster_idx.clear();
	double min_sse = 10000.0;

	/* Creating a 2d cluster values vector */
	 /* Cluster values:
		cluster_vals[i][0] = size of cluster i
		cluster_vals[i][1->dimensions-1] = totals for each dimension of cluster i */
	vector<vector<double>> cluster_vals(clusters, vector<double>(dimensions + 1, 0));

	/* This is where the actual k-means algorithm takes place */
	double sse_total = 0;
	/* Iterating through the whole dataset */
	for (int k = 0; k < observations; k++)
	{
		double min_idx = -1;
		double min_dist = 10000000.0;
		/* argmin algorithm for distance between points and centers */
		for (int l = 0; l < clusters; l++)
		{
			double difference = 0;
			double new_dist = 0;

			/* Iterating through each of the observations' points */
			for (int m = 0; m < dimensions; m++)
			{
				difference = data[k][m] - centers[l][m];
				new_dist += difference * difference;
			}

			if (new_dist < min_dist)
			{
				min_dist = new_dist;
				min_idx = l;
			}
		}
		/* Using a parallel array (vector) to keep track of which cluster each point is in */
		cluster_idx.push_back(min_idx);

		/* Adding to the totals for the center each observation
		   is closest to */
		cluster_vals[min_idx][0] += 1;
		for (int l = 1; l < dimensions + 1; l++)
		{
			cluster_vals[min_idx][l] += data[k][l - 1];
		}

		sse_total += min_dist;
	}
	for (int k = 0; k < clusters; k++)
	{
		for (int l = 1; l < dimensions + 1; l++)
		{
			if (cluster_vals[k][0] == 0)
			{
				centers[k][l - 1] = 0;
			}
			else
			{
				centers[k][l - 1] = (cluster_vals[k][l] / cluster_vals[k][0]);
			}
		}
	}

	/* Need to capture the number of points in each cluster for CH */
	for (int i = 0; i < clusters; i++)
	{
		n_vals.push_back(cluster_vals[i][0]);
	}

	return sse_total;
}

/* The Calinski-Harabasz index function */
double calinskiHarabasz(double sse, int clusters, int observations, int dimensions,
	vector<vector<double>>& data, vector<vector<double>>& centers, vector<double> n)
{
	vector<double> means(dimensions, 0);
	double sb_trace = 0;
	double ch_value = 0;

	/* Calculating the mean centroid over the whole dataset */
	for (int i = 0; i < dimensions; i++)
	{
		for (int j = 0; j < observations; j++)
		{
			means[i] += data[j].at(i);
		}
		means[i] = means[i] / double(observations);
	}

	/* Calculating the trace of the between-cluster scatter matrix */
	for (int i = 0; i < clusters; i++)
	{
		for (int j = 0; j < dimensions; j++)
		{
			sb_trace += n[i] * (centers[i][j] - means[j]) * (centers[i][j] - means[j]);
		}
	}

	/* Calculating the Calinski-Harabasz value */
	ch_value = ((observations - clusters) / (clusters - 1)) * (sb_trace / sse);
	return ch_value;
}

/* The Silhouette Width index function */
double silhouetteWidth(int observations, int dimensions, vector<int>& cluster_idx, vector<vector<double>>& data,
	vector<vector<double>>& centers, int clusters, vector<double> n)
{
	double sw_value = 0;
	double in_mean = 0;
	double out_mean = 0;
	double distance;
	double min_dist;
	int min_idx;

	/* Calculating the in-cluster mean and out-cluster mean with a parallel array (vector) containing cluster indexes */
	for (int i = 0; i < observations; i++)
	{
		for (int j = 0; j < observations; j++)
		{
			if (i == j)
			{
				sw_value += 0;
			}
			/* In-cluster mean */
			else if (cluster_idx[i] == cluster_idx[j])
			{
				for (int k = 0; k < dimensions; k++)
				{
					in_mean += (data[i][k] - data[j][k]) * (data[i][k] - data[j][k]);
				}
			}
		}

		/* Out-cluster mean */
		min_dist = 1000000.0;
		min_idx = 0;
		for (int k = 0; k < clusters; k++)
		{
			distance = 0;
			if (cluster_idx[i] == k)
			{
				distance += 0;
			}
			else
			{
				for (int l = 0; l < dimensions; l++)
				{
					distance += (data[i][l] - centers[k][l]) * (data[i][l] - centers[k][l]);
				}
				if (distance < min_dist)
				{
					min_dist = distance;
					min_idx = k;
				}
			}
		}
		for (int j = 0; j < observations; j++)
		{
			if (cluster_idx[j] == min_idx)
			{
				for (int k = 0; k < dimensions; k++)
				{
					out_mean += (data[i][k] - data[j][k]) * (data[i][k] - data[j][k]);
				}
			}
		}

		/* Preventing a division by zero error */
		if (n[cluster_idx[i]] > 1)
		{
			in_mean = in_mean / (n[cluster_idx[i]] - 1);
		}
		out_mean = out_mean / n[cluster_idx[i]];

		if (in_mean >= out_mean)
		{
			sw_value += (out_mean - in_mean) / in_mean;
		}
		else
		{
			sw_value += (out_mean - in_mean) / out_mean;
		}
	}

	sw_value = sw_value / observations;

	return sw_value;
}