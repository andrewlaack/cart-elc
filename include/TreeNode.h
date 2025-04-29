#include "string"
#include "vector"

struct SplitResults{
	float* XLeft;
	float* XRight;
	int* yLeft;
	int* yRight;
	int leftSize;
	int rightSize;
};

class TreeNode{
	public:
		TreeNode(int classification);

		// featureIndices is an array with size featureCount that
		// specifies which components should be used to generate the split
		//
		// EXAMPLE:
		// [0, 0, 1, 1, 0]
		// 
		// we also need a system of linear equations as the input which
		// will be used to determine the hyperplane that contains them
		// where all coefficients are 0 unless they are included in the
		// features indices list.
		//
		// EXAMPLE:
		// [
		//		[10, 3, 4, 5, 0],
		//		[1, 5, 6, 8, 0],
		// ]
		//

		TreeNode(float* samples, int features, int points, int* indicesOrder, int indicesCount);
		bool isLeaf();
		void setSplit(float splittingValue, int featureIndex);
		float evalSplit(float* X, int* y, int samples, int features, std::string criterion);
		TreeNode* getLeftChild();
		TreeNode* getRightChild();
		void setLeftChild(TreeNode* child);
		void setRightChild(TreeNode* child);
		float getSplitVal();
		int getIndexSplit();
		SplitResults splitOnNode(float* X, int* y, int samples, int features);
		std::string getDotEdges();
		int getClassification();
		float* getEquation();
		bool aboveOrOnPlane(float* sample, int features);
		~TreeNode();

	private:
		bool leaf;
		TreeNode* leftChild = nullptr;
		TreeNode* rightChild = nullptr;
		std::string getDotLabel();
		int classification;
		float* equation = nullptr;
		int* indicesOrder = nullptr;
		int indicesCount;
		float giniImpurity(float* X, int* y, int samples, int features);
		float twoingRule(float* X, int* y, int samples, int features);
		float informationGain(float* X, int* y, int samples, int features);
		float entropy(int* y, int samples);
};


