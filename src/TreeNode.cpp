#include "../include/TreeNode.h"
#include "stdexcept"
#include "math.h"
#include <string>
#include <sstream>
#include "../lib/eigen-3.4.0/Eigen/Dense"
#include "iostream"
#include "unordered_map"
#include "unordered_set"
#include <vector>

TreeNode::TreeNode(int classification){
	leaf = true;
	this->classification = classification;
}

// pass in only the samples you want to split on.
TreeNode::TreeNode(float* samples, int features, int points, int* indicesOrderIn, int indicesCount){

	//std::cout << indicesCount << std::endl;
	//std::cout << features << std::endl;


	this->indicesOrder = new int[indicesCount];
	this->indicesCount = indicesCount;
	
	for(int i = 0 ; i < indicesCount; ++i){
		this->indicesOrder[i] = indicesOrderIn[i];
	}

	//std::cout << std::endl;
	//std::cout << "INDICES: "<< std::endl;
	//for(int i = 0 ; i < indicesCount; ++i){
	//	std::cout << indicesOrder[i] << " "; 
	//}
	//std::cout << std::endl;

    Eigen::MatrixXd data(points, indicesCount);


    for (int i = 0; i < points; i++) {
        for (int j = 0; j < indicesCount; j++) {
            data(i, j) = samples[(i * features) + indicesOrder[j]];
        }
    }

	// uncomment to see input matricies
	//std::cout << data << std::endl << std::endl;;

	// compute means
    Eigen::VectorXd mean = data.colwise().mean();

	// compute difference from mean for each sample
    Eigen::MatrixXd centered = data.rowwise() - mean.transpose();

	// create covariance matrix
    Eigen::MatrixXd covariance = (centered.transpose() * centered) / (points- 1);

	// used to compute eigen vector
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(covariance);

	// eigen vector = normal vector
    Eigen::VectorXd normal = eig.eigenvectors().col(0);

    float offset = mean.dot(normal);

    float* equ = new float[this->indicesCount + 1];
    for (int i = 0; i < indicesCount ; i++) {
        equ[i] = normal[i];
    }

    equ[indicesCount] = offset;

    this->equation = equ;
    this->leaf = false;

}

TreeNode::~TreeNode() {
	if(this->equation != nullptr){
		delete[] this->equation; 
		this->equation = nullptr;
	}
	if(this->indicesOrder != nullptr){
		delete[] this->indicesOrder;
		this->indicesOrder = nullptr;
	}
}

float* TreeNode::getEquation(){
	return this->equation;
}

bool TreeNode::isLeaf(){
	return leaf;
}

// don't include classifications in feature count.
float TreeNode::evalSplit(float* X, int* y, int samples, int features, std::string criterion){

	if(isLeaf()){
		throw std::logic_error("Cannot evaluate split on leaf node.");
	}

	if(criterion == "gini"){

		return giniImpurity(X, y, samples, features);
	}

	if(criterion == "twoing"){
		return twoingRule(X, y, samples, features);
	}
	if(criterion == "information gain"){
		return informationGain(X,y,samples,features);
	}

	throw std::invalid_argument("supported objective functions are 'twoing', 'gini', and 'information gain'");
	return 0.0f;
}


float TreeNode::informationGain(float* X, int* y, int samples, int features) {

	float before = entropy(y, samples);

	std::vector<int> left;
	std::vector<int> right;

	float* currentSample = X;

	for(int i = 0; i < samples; ++i){

		if(aboveOrOnPlane(currentSample, features)){
			left.push_back(y[i]);
		}
		else{
			right.push_back(y[i]);
		}

		currentSample += features;
	}

	float leftEn = entropy(left.data(), left.size());
	float rightEn = entropy(right.data(), right.size());

	float weightedLeft = leftEn * (static_cast<float>(left.size()) / samples);
	float weightedRight = rightEn * (static_cast<float>(right.size()) / samples);

	float after = weightedLeft + weightedRight;

	float final = before - after;

	// we are minimizing objective function thus
	// invert as high entropy is good.
	
	final *= -1;

	return final;
}

float TreeNode::entropy(int* y, int samples){

	if(samples == 0){
		return 0.0f;
	}

    std::unordered_map<int, int> counts;

    for (int i = 0; i < samples; ++i) {
        counts[y[i]]++;
    }

	float entropy = 0.0f;

    for (const auto& entry : counts) {

		float prob = static_cast<float>(entry.second) / static_cast<float>(samples);
		float logProb = std::log2(prob);

		entropy += logProb * prob;
    }

	entropy *= -1;

	return entropy;
}


// outer = (p_L * p_R) * (1/4)
// inner = sum(abs(p(class | t_L) - p(class | t_R))) ^ 2
// final = outer * inner

// note: since the elcclassifier is trying to minimize the objective function,
// but the twoing rule is to be maximized, we multiply this value by -1 before returning.

float TreeNode::twoingRule(float* X, int* y, int samples, int features) {

    std::unordered_map<int, int> ltMap;
    std::unordered_map<int, int> gtMap;

    int ltCount = 0;
    int gteqCount = 0;

    float* currentSample = X;

    for (int i = 0; i < samples; ++i) {
        if (aboveOrOnPlane(currentSample, features)) {
            ltMap[y[i]]++;
            ltCount++;
        } else {
            gtMap[y[i]]++;
            gteqCount++;
        }
        currentSample += features;
    }

    // one side is empty, avoid div by 0.
    if (ltCount == 0 || gteqCount == 0) {
        return 0.0f;
    }

    float pL = static_cast<float>(ltCount) / samples;
    float pR = static_cast<float>(gteqCount) / samples;
    float outer = (pL * pR) * (1 / 4.0f);

    // union labels from both groups in case
	// samples are only in one or the other.
    std::unordered_set<int> classLabels;
    for (const auto& entry : ltMap) {
        classLabels.insert(entry.first);
    }
    for (const auto& entry : gtMap) {
        classLabels.insert(entry.first);
    }

    float inner = 0.0f;
    for (int classLabel : classLabels) {
        int ltClassCount = (ltMap.find(classLabel) != ltMap.end()) ? ltMap[classLabel] : 0;
        float pClassL = static_cast<float>(ltClassCount) / ltCount;

        int gtClassCount = (gtMap.find(classLabel) != gtMap.end()) ? gtMap[classLabel] : 0;
        float pClassR = static_cast<float>(gtClassCount) / gteqCount;

        inner += std::abs(pClassL - pClassR);
    }

    inner = inner * inner;

    // Multiply by -1 because the classifier minimizes the objective function
    float final = -1 * outer * inner;
    return final;
}

float TreeNode::giniImpurity(float* X, int* y, int samples, int features){

	// verified this is perfect. 
	// if there are issues it is with constructor of tree node class. I built it wrong.

	std::unordered_map<int, int> ltMap;
	std::unordered_map<int, int> gtMap;

	int ltCount = 0;
	int gteqCount = 0;

	float* currentSample = X;

	for(int i = 0; i < samples; ++i){

		if(aboveOrOnPlane(currentSample, features)){
			ltMap[y[i]]++;
			ltCount++;
		}
		else{
			gtMap[y[i]]++;
			gteqCount++;
		}

		currentSample += features;
	}

	float ltGini= 1.0f;

	for (const auto& pair : ltMap) {
		ltGini -= pow(float(pair.second) / ltCount, 2);
	}

	float gteqGini = 1.0f;

	for (const auto& pair : gtMap) {
		gteqGini -= pow(float(pair.second) / gteqCount, 2);
	}

	if(gteqCount == 0){
		gteqGini = 0.0f;
	}
	if(ltCount == 0){
		ltGini = 0.0f;
	}


	float gini = gteqGini * float(gteqCount) / samples;
	gini += ltGini * float(ltCount) / samples;

	return gini;
}


// equation form is as follows:
// x, y, z, d
// where ax + by + cz = d


bool TreeNode::aboveOrOnPlane(float* sample, int features){

	float dp = 0;
	for(int i = 0 ; i < indicesCount; ++i){
		dp += sample[indicesOrder[i]] * equation[i];
	}
	float bias = equation[indicesCount];
	float appliedBias = dp - bias;

	// apply slight bias to ensure values on the plane evaluate to |~0|.
	float TOLERANCE = .001;
	appliedBias += TOLERANCE;
	return appliedBias >= 0;

}



void TreeNode::setLeftChild(TreeNode* child){
	leftChild = child;
}

void TreeNode::setRightChild(TreeNode* child){
	rightChild = child;
}

TreeNode* TreeNode::getLeftChild(){
	return leftChild;
}

TreeNode* TreeNode::getRightChild(){
	return rightChild;
}

SplitResults TreeNode::splitOnNode(float* X, int* y, int samples, int features){
	
	int leftSize = 0;
	int rightSize = 0;
	float* current = X;

	// get counts for array init
	for(int i = 0 ; i < samples; ++i){
		if(aboveOrOnPlane(current, features)){
			rightSize += 1;
		}
		else{
			leftSize += 1;
		}

		current += features;
	}
	// done with counting
	
	// init
	SplitResults res;
	res.leftSize = leftSize;
	res.rightSize = rightSize;


	float* right = new float[rightSize * features];
	float* left = new float[leftSize * features];
	int* rightLabels = new int[rightSize];
	int* leftLabels = new int[leftSize];



	int leftFill = 0;
	int rightFill = 0;
	current = X;

    for (int i = 0; i < samples; ++i) {
        if (aboveOrOnPlane(current, features)) {
            for (int x = 0; x < features; ++x) {
                right[rightFill * features + x] = current[x];
            }
            rightLabels[rightFill] = y[i];
            rightFill += 1;
        } else {
            for (int x = 0; x < features; ++x) {
                left[leftFill * features + x] = current[x];
            }
            leftLabels[leftFill] = y[i];
            leftFill += 1;
        }
        current += features;
    }

	res.XLeft = left;
	res.XRight = right;
	res.yRight = rightLabels;
	res.yLeft = leftLabels;

    return res;
}


std::string TreeNode::getDotEdges(){

	if(isLeaf()){
		return "";
	}

	std::string current = getDotLabel() + "->" + leftChild->getDotLabel() + ";\n";
	current += getDotLabel() + "->" + rightChild->getDotLabel() + ";\n";

	current += rightChild->getDotEdges();
	current += leftChild->getDotEdges();

	return current;
}

std::string toStringWithPrecision(float value, int precision = 1) {
    float scale = std::pow(10.0f, precision);
    value = std::round(value * scale) / scale; // Round to the desired precision
    return std::to_string(value).substr(0, std::to_string(value).find(".") + precision + 1);
}

std::string TreeNode::getDotLabel(){
	const void * address = static_cast<const void*>(this);
	std::stringstream ss;
	ss << address;  
	std::string name = ss.str(); 
	if (isLeaf()){
		return "\"" + name + "\nCLASSIFICATION: " + std::to_string(classification) + "\"";
	}

	// equ will always have one more value than the indices.
	// This is because we need an offset (intercept).
	
	std::string equ = "";
	std::string indices = "";
	for(int i = 0 ; i < this->indicesCount; ++i){
		equ += toStringWithPrecision(this->equation[indicesOrder[i]], 2) + " ";
		indices += std::to_string(indicesOrder[i]) + " ";
	}
	equ += toStringWithPrecision(this->equation[this->indicesCount], 2) + " ";

	return "\"" + name + "\nINDICES:\n" + indices + "\nCOEFFICIENTS:\n" + equ + "\"";
}

int TreeNode::getClassification(){
	if(isLeaf()){
		return classification;
	}
	throw std::logic_error("Unable to call getClassification() on internal vertices.");
}
