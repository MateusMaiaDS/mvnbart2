#include<RcppArmadillo.h>
#include<vector>
// Creating the struct
struct Node;
struct modelParam;

struct modelParam {

        arma::mat x_train;
        arma::vec c_train;
        arma::vec q_train;
        arma::mat x_test;

        // BART prior param specification
        int n_tree;
        int d_var; // Dimension of variables in my base
        double alpha;
        double beta;
        double tau_mu;
        double tau_lambda;
        double df_wish;
        arma::mat s_0_wish;
        int node_min_size;

        // Getting the precision parameters from 2d case
        double tau_c;
        double tau_q;
        double rho;

        // MCMC spec.
        int n_mcmc;
        int n_burn;

        // Create an indicator of accepted grown trees
        arma::vec move_proposal;
        arma::vec move_acceptance;


        // Defining the constructor for the model param
        modelParam(arma::mat x_train_,
                   arma::vec c_train_,
                   arma::vec q_train_,
                   arma::mat x_test_,
                   int n_tree_,
                   int node_min_size_,
                   double alpha_,
                   double beta_,
                   double tau_mu_,
                   double tau_lambda_,
                   double df_wish_,
                   arma::mat s_0_wish_,
                   double n_mcmc_,
                   double n_burn_);

};

// Creating a forest
class Forest {

public:
        std::vector<Node*> trees;

        Forest(modelParam &data);
        // ~Forest();
};



// Creating the node struct
struct Node {

     bool isRoot;
     bool isLeaf;
     Node* left;
     Node* right;
     Node* parent;
     arma::vec train_index;
     arma::vec test_index;

     // Branch parameters
     int var_split;
     double var_split_rule;
     double lower;
     double upper;
     double curr_weight; // indicates if the observation is within terminal node or not
     int depth = 0;

     // Leaf parameters
     double mu;
     double lambda;

     // Storing sufficient statistics over the nodes
     double log_likelihood = 0.0;
     double sr_minus_sl = 0.0; // rs = \sum_{i}r_{i} and ls_ = \sum{i}l_{i}
     double sr_minus_sl_sq = 0.0;
     double s_r_minus_l_sq = 0.0; // this is equal to sum_{i}(r_i-l_i)^2;

     double ss_minus_sm = 0.0;     // Same logic from before but r = s
     double ss_minus_sm_sq = 0.0;  // and l = m
     double s_s_minus_m_sq = 0.0;

     double gamma;
     double eta;


     int n_leaf = 0;
     int n_leaf_test = 0;


     // Creating the methods
     void addingLeaves(modelParam& data);
     void deletingLeaves();
     void Stump(modelParam& data);
     void updateWeight(const arma::mat X, int i);
     void getLimits(); // This function will get previous limit for the current var
     void sampleSplitVar(modelParam& data);
     bool isLeft();
     bool isRight();
     void grow(Node* tree, modelParam &data, arma::vec &curr_res);
     void prune(Node* tree, modelParam &data, arma::vec&curr_res);
     void change(Node* tree, modelParam &data, arma::vec&curr_res);
     void nodeUpdateResiduals_c(modelParam& data, arma::vec &curr_res_r, arma::vec& hat_q);
     void nodeLogLike_c(modelParam &data, arma::vec &curr_res_r, arma::vec& hat_q);

     void nodeUpdateResiduals_q(modelParam& data, arma::vec &curr_res_s, arma::vec& hat_c);
     void nodeLogLike_q(modelParam &data, arma::vec &curr_res_s, arma::vec& hat_c);

     void displayCurrNode();

     Node(modelParam &data);
     ~Node();
};

// Creating a function to get the leaves
void leaves(Node* x, std::vector<Node*>& leaves); // This function gonna modify by address the vector of leaves
std::vector<Node*> leaves(Node*x);

