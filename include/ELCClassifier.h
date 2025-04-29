#include "TreeNode.h"
#include <vector>
#include <iostream>
#include "queue"
#include "mutex"

class ELCClassifier{
	public:
		ELCClassifier(int depth, int linearCombinations, int maxThreadCount = 100, std::string objFunction = "gini");
		void fit(float* X, int samples, int* y, int features); 
		int* predict(float* X, int samples, int features);
		std::string getDot();
        ~ELCClassifier(); 
		TreeNode* bestSplit(float* X, int samples, int* y, int features);
		int getSplits();

	private:
		int depth;
		int featureCount = -1;
		int linearCombinations = 0;
		TreeNode* splittingTree = nullptr;
		TreeNode* recurse(float* X, int samples, int* y, int features, int depth);
		int primaryClass(int* y, int labelCount);
		void deleteTree(TreeNode* node);
		TreeNode* bestSplitHelper(float* allSamples, int* y, int sampleCount, int features, std::vector<int> current, int currentIndex, float* bestGini, bool initCall, std::queue<std::vector<int>>& queuedSelections, bool finalPass);
		TreeNode* const bestNodeForSelectSamples(float* allSamples, int* y, int sampleCount, int features, std::vector<int> specifiedSamples, int currentFeature, float* bestGini, std::vector<int> selectedFeatures);
		int maxThreads = 1;
		std::string objectiveFunction = "gini";
		bool homogeneous(int* y, int samples);
};

struct combinations{
	int* combs;
	int rows;
	int columns;
};

