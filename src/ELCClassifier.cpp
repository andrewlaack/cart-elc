#include "../include/ELCClassifier.h"
#include <cmath>
#include <unordered_map>
#include "future"
#include <queue>
#include "map"
#include <stack>

using namespace std;

ELCClassifier::ELCClassifier(int maxDepth, int linearCombinations, int maxThreadCount, std::string objFunction){
	this->depth = maxDepth;
	this->linearCombinations = linearCombinations;
	this->maxThreads = maxThreadCount;
	this->objectiveFunction = objFunction;
}

void ELCClassifier::fit(float* X, int samples, int* y, int features){

	if (splittingTree != nullptr){
		deleteTree(splittingTree);
		splittingTree = nullptr;
	}

	this->featureCount = features;
	this->splittingTree = recurse(X, samples, y, features, depth);
}


std::string ELCClassifier::getDot(){
	if (splittingTree == nullptr){
		throw logic_error("Decision tree must be created prior to generating dot output.");
	}
	std::string edges = splittingTree->getDotEdges();
	std::string dot = "digraph decisionTree {\n" + edges + "}";
	return dot;
}

int ELCClassifier::primaryClass(int* y, int labelCount){
	unordered_map<int,int> map;

	for(int i = 0; i < labelCount; ++i){
		map[y[i]] += 1;
	}

	int mostElements = 0;
	int label = 0;

	for (auto& item : map){
		if(item.second > mostElements){
			mostElements = item.second;
			label = item.first;
		}
	}

	return label;
}

bool ELCClassifier::homogeneous(int* y, int samples){
	
	if(samples == 1){
		return true;
	}

	for(int i = 1 ; i < samples; ++i){
		if(y[i] != y[0]){
			return false;
		}
	}
	return true;
}

TreeNode* ELCClassifier::recurse(float* X, int rows, int* y, int columns, int depthRem){


	for(int i = 0 ; i < rows * columns; ++i){

		if(i % columns == 0){
			std::cout << std::endl;
		}
		std::cout << X[i] << " ";

	}

	std::cout << endl;
	for(int i = 0 ; i < rows; ++i){
		std::cout << y[i] << " ";
	}

	std::cout << endl;

	if(depthRem == 0 || homogeneous(y,rows)){
		TreeNode* ret = new TreeNode(primaryClass(y, rows));
		return ret;
	}


	// found minimum node
	if(rows <= this->linearCombinations){
		TreeNode* ret = new TreeNode(primaryClass(y, rows));
		return ret; 
	}


	// get best split option 
	TreeNode* chosen = bestSplit(X, rows, y, columns);
	SplitResults split = chosen->splitOnNode(X, y, rows, columns);


	//for(int i = 0 ; i < split.leftSize; ++i){
	//	std::cout << split.XLeft[i] << " ";
	//}

	//std::cout << std::endl;

	//for(int i = 0 ; i < linearCombinations; ++i){
	//	std::cout << chosen->getEquation()[i] << " ";
	//}

	//std::cout << std::endl;
	//std::cout << std::endl;

	// no valid splits, but we still did create some new arrays.
	if(split.rightSize == rows || split.leftSize == rows){
		TreeNode* ret = new TreeNode(primaryClass(y, rows));
		
		// line of code is in prison.
		// he cost me ~8 hours of time. Our battle was valiant,
		// but alas you have been found, your leak patched,
		// and you are now under arrest.

		// __________//
		delete chosen;
		chosen = nullptr;
		//^^^^^^^^^^^//

		

		delete[] split.XLeft;
		delete[] split.XRight;
		delete[] split.yLeft;
		delete[] split.yRight;

		split.XLeft = nullptr;
		split.XRight = nullptr;
		split.yLeft = nullptr;
		split.yRight = nullptr;

		return ret; 
	}

	// traverse lt tree
	TreeNode* left = recurse(split.XLeft, split.leftSize, split.yLeft, columns, depthRem - 1);
	// traverse gt tree
	TreeNode* right = recurse(split.XRight, split.rightSize, split.yRight, columns, depthRem - 1);

	chosen->setLeftChild(left);
	chosen->setRightChild(right);

	delete[] split.XLeft;
	delete[] split.XRight;
	delete[] split.yLeft;
	delete[] split.yRight;

	split.XLeft = nullptr;
	split.XRight = nullptr;
	split.yLeft = nullptr;
	split.yRight = nullptr;

	return chosen;
}






// steps:
//
// 1) find all combinations of points
// 		combination count = nCr where n is rows and r is this->linearcombinations
// 2) find all combinations of axis for each combinations
// 		combination count = mCr where m is columns and r is this->linearcombinations
// 3) Evaluate all combinations (impurity)
// 4) Return tree node with the best split.

// TreeNode(float* samples, int features, int points, int* indicesOrder, int indicesCount);

TreeNode* ELCClassifier::bestSplit(float* X, int rows, int* y, int columns) {

//	for(int i = 0 ; i < rows*columns; ++i){
//		if(i % columns == 0){
//			std::cout << std::endl;
//		}
//		std::cout << X[i] << " ";
//
//	}
//	std::cout << std::endl;
//	std::cout << std::endl;

	float bestImpurity = 0.0f;
	float* ptrImpurity = &bestImpurity;
	std::queue<std::vector<int>> queue;
	TreeNode* best = this->bestSplitHelper(X, y, rows, columns, std::vector<int>(), 0, ptrImpurity, true, queue, false);
	return best;
}


// test all combinations of features for given points and return best selection.
//
//
//
//
// this needs to be reworked.
//
// this currently passes in all samples and then finds the best indices to split on instead of what I want
//
// what I should make this do instead is find all indicies and then call a helper method that then computes
// the best points to select with those indices.
//
// To do this I will build another method called bestSplitByPoints which accepts in the indices we are splitting on,
// the other information associated with labels, and all other points for validation. This will then go through all
// combinations of points for the specified indices, returning the best option.


TreeNode* const ELCClassifier::bestNodeForSelectSamples(
float* allSamples, int* y, 
int sampleCount, int features, 
vector<int> specifiedSamples, int currentFeature,
float* bestImpurity,
std::vector<int> selectedFeatures
){

	if((int)selectedFeatures.size() == this->linearCombinations){

		int* featuresToUse = selectedFeatures.data();
		int size = features * specifiedSamples.size();
		float samplesToTest[size];
		int itr = 0;

		for (int x = 0; x < (int)specifiedSamples.size(); ++x) {
			for (int y = 0; y < features; ++y) {
				int sampleIndex = specifiedSamples[x];
				int calculatedIndex = (sampleIndex * features) + y;
				samplesToTest[itr] = allSamples[calculatedIndex];
				itr += 1;

			}
		}

		TreeNode* node = new TreeNode(samplesToTest,  features, this->linearCombinations, featuresToUse, this->linearCombinations);
		*bestImpurity = node->evalSplit(allSamples, y, sampleCount, features, this->objectiveFunction);

		return node;
	}

	if(currentFeature >= features){
		TreeNode* node = nullptr;
		*bestImpurity = INFINITY;
		return node;
	}
	
	// without this one included
	
	float left = 0;
	float right = 0;
	float* leftPtr = &left;
	float* rightPtr = &right;

	TreeNode* bestWithout = bestNodeForSelectSamples(allSamples, y, sampleCount, features, specifiedSamples, currentFeature + 1, leftPtr, selectedFeatures);

	// with this one included
	
	selectedFeatures.push_back(currentFeature);


	TreeNode* bestWith = bestNodeForSelectSamples(allSamples, y, sampleCount, features, specifiedSamples, currentFeature + 1, rightPtr, selectedFeatures);

	if(*leftPtr > *rightPtr){
		if(bestWithout != nullptr){
			delete bestWithout;
			bestWithout = nullptr;
		}
		*bestImpurity = *rightPtr;
		return bestWith;
	}
	else{
		if(bestWith != nullptr){
			delete bestWith;
			bestWith = nullptr;
		}
		*bestImpurity = *leftPtr;
		return bestWithout;
	}


}

// init call is used to ensure we clean up the queue
TreeNode* ELCClassifier::bestSplitHelper(float* allSamples, int* y, int sampleCount, int features, vector<int> current, int currentFeature, float* bestImpurity, bool initCall, std::queue<std::vector<int>>& queuedSelections, bool finalPass) {

	if((int)current.size() == this->linearCombinations || finalPass){

		// this will be -1 when calling for the final time if it is the init call. 
		// This is messy, but I don't know how to make it better.

		if(!finalPass){
			queuedSelections.push(current);
		}

		// this is the only location where we evaluate potential samples to split on.
		if((int)queuedSelections.size() > this->maxThreads or finalPass){

			float currentBestImpurity = INFINITY;
			TreeNode* bestNode = nullptr;

    		std::vector<std::future<TreeNode*>> futureList;
			std::vector<float> floats = std::vector<float>();
			for(int i = 0 ; i < (int)queuedSelections.size(); ++i){
				floats.push_back(INFINITY);
			}

		
			int itr = 0;
			while ((int)queuedSelections.size() > 0) {

				float* tempImpurity = &floats[itr];
				auto currentVec = queuedSelections.front();
				queuedSelections.pop();

				std::future<TreeNode*> futureNode = std::async(std::launch::async, 
					&ELCClassifier::bestNodeForSelectSamples, 
					*this,  
					allSamples, 
					y, 
					sampleCount, 
					features, 
					currentVec, 
					0, 
					tempImpurity, 
					std::vector<int>());

				futureList.push_back(std::move(futureNode));
				itr++;
			}

		itr = 0;

		for (auto& future : futureList) {
			TreeNode* currentNode = future.get();  // This blocks until the future is ready
			float tempImpurity = floats[itr];

			if (tempImpurity < currentBestImpurity) {
				currentBestImpurity = tempImpurity;
				
				// Delete previous best node
				if (bestNode != nullptr) {
					delete bestNode;
					bestNode = nullptr;
				}

				bestNode = currentNode;
			} else {
				delete currentNode;
				currentNode = nullptr;
			}

			itr++;
		}

		*bestImpurity = currentBestImpurity;
		return bestNode;

		}
		else{
			*bestImpurity = INFINITY;
			return nullptr;
		}
	}

	if(currentFeature >= sampleCount){
		TreeNode* node = nullptr;
		*bestImpurity = INFINITY;
		return node;
	}


	// without this one included
	
	float left = 0;
	float right = 0;
	float* leftPtr = &left;
	float* rightPtr = &right;


	TreeNode* bestWithout = bestSplitHelper(allSamples, y, sampleCount, features, current, currentFeature + 1, leftPtr, false, queuedSelections, false);

	// with this one included
	current.push_back(currentFeature);


	TreeNode* bestWith = bestSplitHelper(allSamples, y, sampleCount, features, current, currentFeature + 1, rightPtr, false, queuedSelections, false);
	
	// this is used to ensure that even if the total number of evaluated splits is less than the number of allowed threads
	// we still clear out the queue.
	
	if(initCall){
		float curImpurity = INFINITY;
		TreeNode* final = bestSplitHelper(allSamples, y, sampleCount, features, current, currentFeature + 1, &curImpurity, false, queuedSelections, true);
		if(curImpurity < left){

			leftPtr = &curImpurity;

			if(bestWithout != nullptr){
				delete bestWithout;
				bestWithout = nullptr;
			}

			bestWithout = final;
			final = nullptr;
		}
	}

	if(*leftPtr > *rightPtr){
		if(bestWithout != nullptr){
			delete bestWithout;
			bestWithout = nullptr;
		}
		*bestImpurity = *rightPtr;
		return bestWith;
	}
	else{
		if(bestWith != nullptr){
			delete bestWith;
			bestWith = nullptr;
		}
		*bestImpurity = *leftPtr;
		return bestWithout;
	}
}

int* ELCClassifier::predict(float* X, int samples, int features) {

	if(featureCount == -1){
		throw logic_error("Unable to predict prior to calling fit().");
	}

	if(features != this->featureCount){
		throw invalid_argument("Incorrect number of features for prediction.");
	}

	int* predictions = new int[samples];

	for(int i = 0; i < samples; ++i){
		TreeNode* current = splittingTree;
		while(!current->isLeaf()){
			float* currentElement = X;
			currentElement += features * i;
			bool above = current->aboveOrOnPlane(currentElement, features);
			if(above){
				current = current->getRightChild();
			}
			else{
				current = current->getLeftChild();
			}
		}
		predictions[i] = current->getClassification();
	}

	return predictions;
}

ELCClassifier::~ELCClassifier(){

	if(this->splittingTree != nullptr){
		deleteTree(this->splittingTree);
		this->splittingTree = nullptr;
	}

}

void ELCClassifier::deleteTree(TreeNode* node){

	if(node == nullptr){
		return;
	}

	if(node->getLeftChild() != nullptr){
		deleteTree(node->getLeftChild());
		node->setLeftChild(nullptr);
	}

	if(node->getRightChild() != nullptr){
		deleteTree(node->getRightChild());
		node->setRightChild(nullptr);
	}

	delete node;
}

int ELCClassifier::getSplits(){
	
	TreeNode* current = splittingTree;

	if(current == nullptr){
		return 0;
	}


    int count = 0;
    std::stack<TreeNode*> stack;
    stack.push(splittingTree);

    while (!stack.empty()) {
        TreeNode* current = stack.top();
        stack.pop();

        if (!current->isLeaf()) {
            count++; 
            if (current->getLeftChild()) stack.push(current->getLeftChild());
            if (current->getRightChild()) stack.push(current->getRightChild());
        }
    }

    return count;
}

