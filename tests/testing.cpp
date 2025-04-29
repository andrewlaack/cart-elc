#include <random>
#include "iostream"
#include "../include/ELCClassifier.h"


void testHyperplaneCreationAndEvaluation(){

	// test square matricies 100 times each
	// for size 1-20.

	for(int i = 0 ; i < 2100; ++i){
		int features = int(i / 100) + 1;
		int points = features;
		float samples[features * features];

		std::uniform_int_distribution<> dist(0, 99);  // Uniform distribution between 0 and 99
		std::random_device rd;  // Get a random seed from hardware
		for(int i = 0 ; i < features * features ; ++i){
			samples[i] = dist(rd);
		}

		int indicesOrder[features - 1];
		for(int i = 0 ; i < features; ++i){
			indicesOrder[i] = i;
		}

		TreeNode node = TreeNode(samples, features, points, indicesOrder, features);
		
		float* equ = node.getEquation();

		// evaluate to verify that all samples are on the plane
		for(int x = 0; x < points; ++x){
			float result = 0;

			for(int i = 0 ; i < features; ++i){
				result += equ[i] * samples[i + x*features];
			}

			// some margin of error; these are floats after all.
			if(fabs(result - equ[features]) > .001){
				throw std::logic_error("Problem with hyperplane equation computation.");
			}
		}
	}
	std::cout << "Hyperplane equations computed as expected" << std::endl;


	// validate that it can take in indicesCounts and orders that aren't the same
	// as the features. 
	//
	// Instead of enforcing that we have square matricies, we specify which features we care about right now.
	// This means the indices count should still be equal to the points, but still.
	//
	// This is useful for evaluating splits with respect to the hyperplane because we can easily understand 
	// which features we are doing the linear combination of.
	
	float samples[] = {
		22, 2, 2, 8, 4,
		40, 4, 5, 4, 1
	};

	int points = 2;
	int indicesOrder[2] = {1,2};
	int features = 5;
	int indicesCount = 2;
	TreeNode node = TreeNode(samples, features, points, indicesOrder, indicesCount);

	// we already know this works because of the above test
	float samples1[] = {
		2, 2, 
		4, 5,
	};
	int points1 = 2;
	int indicesOrder1[2] = {0,1};
	int features1 = 2;
	int indicesCount1= 2;
	TreeNode node1 = TreeNode(samples1, features1, points1, indicesOrder1, indicesCount1);

	// by verifying these are the same, we have shown that the hyperplane respects the indices
	// we specified.

	for(int i = 0 ; i < indicesCount1 + 1; ++i){
		if(node1.getEquation()[i] != node.getEquation()[i]){
			throw std::logic_error("Indices order or count not working as expected.");
		}
	}
	std::cout << "Verified indices order and count work as expected" << std::endl;

	// equation for plane is:
	//-0.83205 = a
	//0.5547 = b
	//-0.5547 = d
	
	int numSamples = 6;

	// these can be evaluated manually with the equation shown above and the formula:
	// ax + by = d
	// WHERE:
	// x = first feature
	// y = second feature

	float evalSamples[] = {
		2, 2, // on plane
		4, 5, // on plane
		2.1, 2,// below
		2, 2.1,// above
		10, 13.99,// below
		10, 14, // on plane 
	};

	float* eval = evalSamples;

	bool answers[] = {
		true,
		true,
		false,
		true,
		false,
		true,
	};

	for(int i = 0 ; i < numSamples; ++i){

		if(node1.aboveOrOnPlane(eval, 2) != answers[i]){
			throw std::logic_error("Above or on plane not working as expected.");
		}
		// get next sample
		eval += 2;
	}

	std::cout << "Verified evaluation for point on or above works as expected." << std::endl;
}




void testInformationGain(){

	// this should result in x = 1.
	float train[] = {
		1
	};

	float validation[] = {
		// these three are left
		0, 5, 2,
		0, 6, 3,
		0, 7, 4,

		// these three are right
		4, 8, 5,
		5, 9, 6,
		6, 10, 7
	};

	int validationClasses[] = {
		// these three are left
		0,
		0,
		1,

		// these three are right
		1,
		0,
		1
	};

	// before = [0, 0, 1, 1, 0, 1]
	// left = [0,0,1]
	// right = [1,0,1]
	//
	// before = -(.5 log_2 .5 + .5 log_2 .5)
	// = 1
	//
	// left H(X) = 0.9183
	// right H(X) = 0.9183
	//
	// weighted = .9183
	//
	// IG = before - after
	// = .0817 (bigger is better)
	// 
	// INV(IG) = .0817 * -1
	// = -.0817


	int points1 = 1;
	int indicesOrder1[1] = {0};
	int features1 = 1;
	int indicesCount1= 1;

	TreeNode node1 = TreeNode(train, features1, points1, indicesOrder1, indicesCount1);

	float infGain = node1.evalSplit(validation, validationClasses, 6, 3, "information gain");
	
	// we return negative as the rest of our logic is set up to minimize
	// the objetive function
	
	std::cout << "FINAL EVAL: " << infGain << std::endl;

	float expected = -.0817f;
	float diff = expected - infGain;

	diff = fabs(diff);

	std::cout << "DIFF: " << diff << std::endl;

	if(diff > 0.0001f){
		throw std::logic_error("information gain not working as expected");
	}

	std::cout << "Verified information gain computed correctly" << std::endl;
}

void testTwoing(){

	// this should result in x = 1.
	float train[] = {
		1
	};

	float validation[] = {
		// these three are left
		0, 5, 2,
		0, 6, 3,
		0, 7, 4,

		// these three are right
		4, 8, 5,
		5, 9, 6,
		6, 10, 7
	};

	int validationClasses[] = {
		// these three are left
		0,
		0,
		1,

		// these three are right
		1,
		0,
		1
	};

	// left = [0,0,1]
	// right = [1,0,1]
	//
	// p_L = .5
	// p_R = .5
	// p_L * p_R = .25
	// (p_L * p_R) / 4 = .0625 = outer
	//
	// p(0 | t_L) = 2/3
	// p(1 | t_L) = 1/3
	//
	// p(0 | t_R) = 1/3
	// p(1 | t_R) = 2/3
	//
	// 2/3 - 1/3 = 1/3
	// 1/3 - 2/3 = -1/3
	//
	// 1/3 + 1/3 = 2/3
	// 
	// 2/3 ^ 2 = 4/9 = inner
	//
	// 4/9 * .0625 = .02777777777

	int points1 = 1;
	int indicesOrder1[1] = {0};
	int features1 = 1;
	int indicesCount1= 1;

	TreeNode node1 = TreeNode(train, features1, points1, indicesOrder1, indicesCount1);

	float twoingEval = node1.evalSplit(validation, validationClasses, 6, 3, "twoing");
	
	// we return negative as the rest of our logic is set up to minimize
	// the objetive function
	
	twoingEval *= -1;
	std::cout << "FINAL EVAL: " << twoingEval << std::endl;

	float expected = 0.0277777f;
	// remember this returns the negative twoing
	float diff = twoingEval - expected;

	diff = fabs(diff);

	std::cout << "DIFF: " << diff << std::endl;

	if(diff > 0.0001f){
		throw std::logic_error("twoing rule not working as expected");
	}

	std::cout << "Verified twoing rule computed correctly" << std::endl;
}


void testGini(){


	// this should result in x = 1.
	float train[] = {
		1
	};


	// we notice that all of the ones classified with 0 as the y value
	// have x > 2 and the one with classification 1 has x < 2.

	// as such, the gini impurity should be 0 (best).
	
	float validation[] = {
		0, 2, 5,
		4, 5, 5,
		4, 5, 5,
		4, 5, 5,
		4, 5, 5,
	};

	int validationClasses[] = {
		1,
		0,
		0,
		0,
		0
	};

	int points1 = 1;
	int indicesOrder1[1] = {0};
	int features1 = 1;
	int indicesCount1= 1;

	TreeNode node1 = TreeNode(train, features1, points1, indicesOrder1, indicesCount1);

	float giniImpurity = node1.evalSplit(validation, validationClasses, 5, 3, "gini");

	if(giniImpurity != 0.0f){
		throw std::logic_error("Gini impurity not working as expected");
	}

	// so long as len >= 1 we are good to evaluate
	float val2[] = {
		10, 2,
		10, 2,
		0, 2,
		0, 10
	};

	// this means we should have one of each below and one of each above.
	// as such:
	// 1 - (.5)^2 = .75
	// .75 - (.5)^2 = .5
	//
	// This is what we should find for both sides. Then since they are equally weighted, the final
	// impurity should be .5

	int val2Classes[] = {
		0,
		1,
		0,
		1
	};

	float gini = node1.evalSplit(val2, val2Classes, 4, 2, "gini");
	if(gini != .5f){
		throw std::logic_error("Gini impurity not working as expected");
	}

	// now let's verify weighting works as expected.

	// lt split:
	// 3, 2, 5, 7, 7
	//
	// 1/5^2 + 1/5^2 + 1/5^2 + 2/5^2 = ~.28
	// gini for lt split = 1 - .28 = .72
	//
	// gt split:
	// 1, 0
	// GT Split gini = .5
	//
	// Weighted gini:
	//
	// (2 * .5) + (5*.72) = 4.6
	// 4.6 / 7 = ~.657
	
	float val3[] = {
		0,
		0,
		0,
		0,
		0,
		2,
		1
	};

	int val3Classes[] = {
		// less than below
		3,
		2,
		5,
		7,
		7,
		// greater than below
		1,
		0
	};

	float finalGini = node1.evalSplit(val3, val3Classes, 7, 1, "gini");
	float expectedGini = .657142857;


	float diff = finalGini - expectedGini;
	if(diff > .001 || diff < -.001){
		throw std::logic_error("Gini impurity not working as expected");
	}
	
	std::cout << "Verified gini impurity computed correctly" << std::endl;
}

void testSplitting(){

	int points1 = 1;
	int indicesOrder1[1] = {0};
	int features1 = 1;
	int indicesCount1= 1;
	float train[] = {
		1.0f
	};

	TreeNode node1 = TreeNode(train, features1, points1, indicesOrder1, indicesCount1);

	// x >= 1
	float val[] = {
		0.0f, 2.0f,
		0.0f, 2.0f,
		0.0f, 5.0f,
		0.0f, 7.0f,
		0.0f, 8.0f,
		1.0f, 9.0f,
		7.0f, 10.0f
	};

	int valClasses[] = {
		3,
		2,
		5,
		7,
		7,
		// greater than below
		1,
		0
	};

	int featureCount = 2;
	int sampleCount = 7;

	SplitResults res = node1.splitOnNode(val, valClasses, sampleCount, featureCount);



	int expLeftClasses[] = {
		3,
		2,
		5,
		7,
		7
	};

	int expRightClasses[] = {
		1,
		0
	};

	float expLeftSamples[] = {
		0, 2,
		0, 2,
		0, 5,
		0, 7,
		0, 8,
	};

	float expRightSamples[] = {
		1, 9,
		7, 10
	};

	// verify proper splitting of labels.

	for(int i = 0 ; i < 5; ++i){
		if(res.yLeft[i] != expLeftClasses[i]){
			throw new std::logic_error("Splitting not working as expected.");
		}
	}

	for(int i = 0 ; i < 2; ++i){
		if(res.yRight[i] != expRightClasses[i]){
			throw new std::logic_error("Splitting not working as expected.");
		}
	}

	for(int i = 0 ; i < 2; ++i){
		for(int x = 0 ; x < 2; ++x){
			if(res.XRight[i*2 + x] != expRightSamples[i*2 + x]){
				throw new std::logic_error("Splitting not working as expected.");
			}
		}
	}

	for(int i = 0 ; i < 5; ++i){
		for(int x = 0 ; x < 2; ++x){
			if(res.XLeft[i*2 + x] != expLeftSamples[i*2 + x]){
				throw new std::logic_error("Splitting not working as expected.");
			}
		}
	}

	delete[] res.XRight;
	delete[] res.XLeft;
	delete[] res.yRight;
	delete[] res.yLeft;

	std::cout << "Verified splitting on tree nodes works as expected." << std::endl;
}

void testBestSplit(){

	
	float val[] = {

		4.1f,  4.5f,  4.5f,  4.2f,  4.4f,  7.1f,  5.5f,  2.2f,  9.4f,  1.2f,
		5.5f,  2.2f,  9.4f,  4.0f,  1.25f, 3.5f,  4.1f,  4.5f,  4.5f,  4.2f,
		4.4f,  7.1f,  5.5f,  2.2f,  9.4f,  1.2f,  5.5f,  2.2f,  9.4f,  4.0f,
		1.25f, 3.5f,  4.1f,  4.5f,  4.5f,  4.2f,  4.4f,  7.1f,  5.5f,  2.2f,
		9.4f,  1.2f,  5.5f,  2.2f,  9.4f,  4.0f,  1.25f, 3.5f,  4.1f,  4.5f

	};


	// 0 1 2
	// 0 1 2
	
	int valClasses[] = {
		0,
		1,
		0,
		0,
		2
	};

	int MAX_DEPTH = 1;
	int LINEAR_COMBINATIONS = 3;
	int featureCount = 10;
	int sampleCount = 5;


	ELCClassifier clf(MAX_DEPTH, LINEAR_COMBINATIONS);
	std::vector<int> input = std::vector<int>();
	TreeNode* split = clf.bestSplit(val, sampleCount, valClasses, featureCount);

	float value = split->evalSplit(val, valClasses, sampleCount, featureCount, "gini");

	delete split;
	


	// still need to add proper validation.

	std::cout << "Verified best by points computes without errors." << std::endl;

}

void testFit(){
	float val[] = {
		6.83f, 5.94f, 7.06f, 3.46f, 7.79f, 9.11f, 4.73f, 7.32f, 5.98f, 0.09f, 
		9.93f, 1.45f, 6.1f, 6.72f, 1.36f, 7.3f, 0.52f, 9.62f, 0.98f, 1.89f, 
		2.82f, 1.78f, 7.21f, 4.6f, 8.54f, 3.48f, 9.18f, 3.64f, 0.8f, 5.4f, 
		6.31f, 9.6f, 9.37f, 6.15f, 9.36f, 9.53f, 7.54f, 0.63f, 0.77f, 2.67f, 
		5.71f, 1.74f, 0.62f, 0.34f, 7.39f, 1.05f, 3.97f, 3.66f, 2.99f, 8.95f, 
		6.44f, 3.39f, 7.11f, 5.85f, 7.68f, 9.16f, 6.73f, 1.69f, 6.91f, 1.54f, 
		9.93f, 6.32f, 4.66f, 3.22f, 2.17f, 6.22f, 3.04f, 6.83f, 5.7f, 1.22f, 
		5.31f, 4.43f, 0.93f, 3.89f, 6.92f, 5.31f, 2.05f, 5.73f, 5.83f, 1.62f, 
		2.14f, 0.43f, 3.81f, 5.4f, 0.07f, 7.11f, 4.94f, 3.3f, 4.04f, 4.7f, 
		2.0f, 1.63f, 7.04f, 3.45f, 1.72f, 2.1f, 1.83f, 8.61f, 0.21f, 5.11f, 
		7.63f, 4.94f, 4.69f, 6.11f, 3.44f, 2.91f, 5.12f, 9.61f, 9.43f, 9.64f, 
		6.17f, 8.1f, 4.18f, 3.57f, 3.02f, 4.94f, 6.52f, 9.97f, 0.68f, 4.64f, 
		2.29f, 7.01f, 1.31f, 3.47f, 5.54f, 8.22f, 7.63f, 2.42f, 8.67f, 1.8f, 
		5.23f, 2.3f, 9.51f, 8.93f, 1.75f, 1.5f, 1.04f, 0.24f, 3.26f, 8.5f, 
	};


	// 0 1 2
	// 0 1 2
	
	int valClasses[] = {
		0,
		1,
		10,
		7,
		7,
		0,
		6,
		4,
		10,
		5,
		8,
		6,
		9,
		9,
	};

	int MAX_DEPTH = 50;
	int LINEAR_COMBINATIONS = 2;
	int featureCount = 10;
	int sampleCount = 14;

	ELCClassifier clf = ELCClassifier(MAX_DEPTH, LINEAR_COMBINATIONS);

	clf.fit(val, sampleCount, valClasses, featureCount);
	std::cout << "Fitting runs successfully" << std::endl;

}


void getDot(){

	float val[] = {
		1.2f, 2.0f,
		1.0f, 2.0f,
		1.0f, 2.1f,
		1.2f, 9.0f,
		5.2f, 8.0f,
		3.2f, 2.0f,
		4.2f, 7.0f,
		0.2f, 7.0f,
		1.0f, 2.0f,
		0.0f, 7.1f,
        2.3f, 6.4f,
        8.1f, 3.5f,
        7.2f, 4.3f,
        3.9f, 1.5f,
        5.0f, 6.7f,
	};


	// 0 1 2
	// 0 1 2
	
	int valClasses[] = {
		0,
		1,
		10,
		1,
		1,
		5,
		7,
		8,
		6,
		9,
		5,
		7,
		8,
		6,
		9,
	};

	int MAX_DEPTH = 50;
	int LINEAR_COMBINATIONS = 1;
	int featureCount = 2;
	int sampleCount = 15;


	ELCClassifier clf = ELCClassifier(MAX_DEPTH, LINEAR_COMBINATIONS);
	clf.fit(val, sampleCount, valClasses, featureCount);
	std::cout << clf.getDot();
	std::cout << std::endl;
}

void testPrediction(){

	float val[] = {
		1.2f, 9.0f,
		5.2f, 8.0f,
		3.2f, 2.0f,
		4.2f, 7.0f,
		0.2f, 7.0f,
		1.0f, 2.0f,
		0.0f, 7.1f,
        2.3f, 6.4f,
        8.1f, 3.5f,
        7.2f, 4.3f,
        3.9f, 1.5f,
        5.0f, 6.7f,
        4.8f, 9.1f,
        2.1f, 8.0f,
        1.4f, 3.3f,
        6.2f, 5.5f,
        7.9f, 2.4f,
        0.9f, 8.3f,
        4.4f, 1.8f,
        3.0f, 6.1f,
        2.8f, 7.9f,
        5.3f, 4.2f
	};

	int valClasses[] = {
		1,
		1,
		5,
		7,
		8,
		6,
		9,
		5,
		7,
		8,
		6,
		9,
		0,
		1,
		5,
		7,
		8,
		6,
		9,
		10,
		2,
		1
	};

	int MAX_DEPTH = 50;
	int LINEAR_COMBINATIONS = 1;
	int featureCount = 2;
	int sampleCount = 22;

	ELCClassifier clf = ELCClassifier(MAX_DEPTH, LINEAR_COMBINATIONS);
	clf.fit(val, sampleCount, valClasses, featureCount);

	int* preds = clf.predict(val, sampleCount, featureCount);
	for(int i = 0 ; i < sampleCount; ++i){
		if(preds[i] != valClasses[i]){
			throw std::logic_error("Computing axis splits (LC=1) not working properly.");
		}

	}
	delete[] preds;

	std::cout << "Verified axis splits are working as expected" << std::endl;
}

void getRNDTree(int featureCount, int sampleCount, int MAX_DEPTH, int LINEAR_COMBINATIONS) {

    std::vector<float> val(featureCount * sampleCount);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(1.0f, 5.0f); // Random values between 1.0 and 5.0

    for (auto& v : val) {
        v = dist(gen);
    }

    // Generate random class labels
    std::vector<int> valClasses(sampleCount);
    std::uniform_int_distribution<int> classDist(0, 10); // Random classes between 0 and 10

    for (auto& c : valClasses) {
        c = classDist(gen);
    }

    // Create and fit the classifier
    ELCClassifier clf = ELCClassifier(MAX_DEPTH, LINEAR_COMBINATIONS, 50, "twoing");

	for(int i = 0 ; i < 2; ++i){
    	clf.fit(val.data(), sampleCount, valClasses.data(), featureCount);
		std::cout << std::endl;
		std::cout << std::endl;
	}
    // std::cout << std::endl;
    // std::cout << std::endl;
    // std::cout << std::endl;
    // std::cout << std::endl;
    // 
	// std::cout << clf.getDot() << std::endl;

	// std::cout << std::endl;
	// 

	// std::cout << "SPLITS: "<< clf.getSplits() << std::endl;

}

// TODO:
// x Implement split on node
// x Build logic for fitting :)
// x Build logic for prediction
// x Build dot logic for graphing
// x Multicore support
// - More tests
// - Benchmarking

//==3641677== 
//==3641677== HEAP SUMMARY:
//==3641677==     in use at exit: 0 bytes in 0 blocks
//==3641677==   total heap usage: 652,186 allocs, 652,186 frees, 28,881,639 bytes allocated
//==3641677== 
//==3641677== All heap blocks were freed -- no leaks are possible
//==3641677== 
//==3641677== For lists of detected and suppressed errors, rerun with: -s
//==3641677== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
// haha



int main(){

	//testHyperplaneCreationAndEvaluation();
	testGini();
	testTwoing();
	testInformationGain();
	//testSplitting();
	//testBestSplit();
	//testFit();
	//getDot();
	//testPrediction();

	//getRNDTree(3, 10, 10, 2);
	return 0;
}
