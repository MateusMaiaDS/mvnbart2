#include "mvnbart.h"
#include <iomanip>
#include<cmath>
#include <random>
#include <RcppArmadillo.h>
using namespace std;

// =====================================
// Statistics Function
// =====================================


void printMatrix(const arma::mat& M) {
        int rows = M.n_rows;
        int cols = M.n_cols;

        std::cout << "Matrix:\n";
        std::cout << std::fixed << std::setprecision(5);

        int max_rows = std::min(rows, 5);
        int max_cols = std::min(cols, 5);

        for (int i = 0; i < max_rows; i++) {
                for (int j = 0; j < max_cols; j++) {
                        std::cout << std::setw(10) << M(i, j) << " ";
                }
                std::cout << "\n";
        }
}


// [[Rcpp::export]]
double gamma_pdf(double x, double a, double b) {

        double gamma_fun = tgamma(a);
        if(isinf(gamma_fun)){
                return 0.0;
        } else {
                return (pow(x, a-1) * exp(-x*b)*pow(b,a)) / ( gamma_fun);
        }


}

// [[Rcpp::export]]
double r_gamma_pdf(double x, double a, double b) {

        return R::dgamma(x,a,1/b,false);

}

// [[Rcpp::export]]
void print_mat_subset(arma::mat X) {
        int n_rows = X.n_rows;
        int n_cols = X.n_cols;

        // print the first 5 rows and 5 columns
        for (int i = 0; i < n_rows; i++) {
                if (i >= 5) break; // only print first 5 rows
                for (int j = 0; j < n_cols; j++) {
                        if (j >= 5) break; // only print first 5 columns
                        Rcpp::Rcout << std::setw(10) << X(i, j) << " ";
                }
                Rcpp::Rcout << std::endl;
        }
}


// Calculating the log-density of a MVN(0, Sigma)
//[[Rcpp::export]]
double log_dmvn(arma::vec& x, arma::mat& Sigma){

        arma::mat L = arma::chol(Sigma ,"lower"); // Remove diagonal later
        arma::vec D = L.diag();
        double p = Sigma.n_cols;

        arma::vec z(p);
        double out;
        double acc;

        for(int ip=0;ip<p;ip++){
                acc = 0.0;
                for(int ii = 0; ii < ip; ii++){
                        acc += z(ii)*L(ip,ii);
                }
                z(ip) = (x(ip)-acc)/D(ip);
        }
        out = (-0.5*sum(square(z))-( (p/2.0)*log(2.0*M_PI) +sum(log(D)) ));


        return out;

};

// //[[Rcpp::export]]
arma::mat sum_exclude_col(arma::mat mat, int exclude_int){

        // Setting the sum matrix
        arma::mat m(mat.n_rows,1);

        if(exclude_int==0){
                m = sum(mat.cols(1,mat.n_cols-1),1);
        } else if(exclude_int == (mat.n_cols-1)){
                m = sum(mat.cols(0,mat.n_cols-2),1);
        } else {
                m = arma::sum(mat.cols(0,exclude_int-1),1) + arma::sum(mat.cols(exclude_int+1,mat.n_cols-1),1);
        }

        return m;
}



// Initialising the model Param
modelParam::modelParam(arma::mat x_train_,
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
                       double n_burn_){


        // Assign the variables
        x_train = x_train_;
        c_train = c_train_;
        q_train = q_train_;
        x_test = x_test_;
        n_tree = n_tree_;
        node_min_size = node_min_size_;
        alpha = alpha_;
        beta = beta_;
        tau_mu = tau_mu_;
        tau_lambda = tau_lambda_;
        df_wish = df_wish_;
        s_0_wish = s_0_wish_; // Precision matrix prior;
        n_mcmc = n_mcmc_;
        n_burn = n_burn_;

        // Grow acceptation ratio
        move_proposal = arma::vec(3,arma::fill::zeros);
        move_acceptance = arma::vec(3,arma::fill::zeros);

}

// Initialising a node
Node::Node(modelParam &data){
        isLeaf = true;
        isRoot = true;
        left = NULL;
        right = NULL;
        parent = NULL;
        train_index = arma::vec(data.x_train.n_rows,arma::fill::ones)*(-1);
        test_index = arma::vec(data.x_test.n_rows,arma::fill::ones)*(-1) ;

        var_split = -1;
        var_split_rule = 0.0;
        lower = 0.0;
        upper = 1.0;
        mu = 0.0;
        n_leaf = 0.0;
        n_leaf_test = 0;
        log_likelihood = 0.0;
        depth = 0;
}

Node::~Node() {
        if(!isLeaf) {
                delete left;
                delete right;
        }
}

// Initializing a stump
void Node::Stump(modelParam& data){

        // Changing the left parent and right nodes;
        left = this;
        right = this;
        parent = this;
        // n_leaf  = data.x_train.n_rows;

        // Updating the training index with the current observations
        for(int i=0; i<data.x_train.n_rows;i++){
                train_index[i] = i;
        }

        // Updating the same for the test observations
        for(int i=0; i<data.x_test.n_rows;i++){
                test_index[i] = i;
        }

}

void Node::addingLeaves(modelParam& data){

     // Create the two new nodes
     left = new Node(data); // Creating a new vector object to the
     right = new Node(data);
     isLeaf = false;

     // Modifying the left node
     left -> isRoot = false;
     left -> isLeaf = true;
     left -> left = left;
     left -> right = left;
     left -> parent = this;
     left -> var_split = 0;
     left -> var_split_rule = 0.0;
     left -> lower = 0.0;
     left -> upper = 1.0;
     left -> mu = 0.0;
     left -> log_likelihood = 0.0;
     left -> n_leaf = 0.0;
     left -> depth = depth+1;
     left -> train_index = arma::vec(data.x_train.n_rows,arma::fill::ones)*(-1);
     left -> test_index = arma::vec(data.x_test.n_rows,arma::fill::ones)*(-1);

     right -> isRoot = false;
     right -> isLeaf = true;
     right -> left = right; // Recall that you are saving the address of the right node.
     right -> right = right;
     right -> parent = this;
     right -> var_split = 0;
     right -> var_split_rule = 0.0;
     right -> lower = 0.0;
     right -> upper = 1.0;
     right -> mu = 0.0;
     right -> log_likelihood = 0.0;
     right -> n_leaf = 0.0;
     right -> depth = depth+1;
     right -> train_index = arma::vec(data.x_train.n_rows,arma::fill::ones)*(-1);
     right -> test_index = arma::vec(data.x_test.n_rows,arma::fill::ones)*(-1);


     return;

}

// Creating boolean to check if the vector is left or right
bool Node::isLeft(){
        return (this == this->parent->left);
}

bool Node::isRight(){
        return (this == this->parent->right);
}

// Sample var
void Node::sampleSplitVar(modelParam &data){

          // Sampling one index from 0:(p-1)
          var_split = arma::randi(arma::distr_param(0,(data.x_train.n_cols-1)));

}
// This functions will get and update the current limits for this current variable
void Node::getLimits(){

        // Creating  a new pointer for the current node
        Node* x = this;
        // Already defined this -- no?
        lower = 0.0;
        upper = 1.0;
        // First we gonna check if the current node is a root or not
        bool tree_iter = x->isRoot ? false: true;
        while(tree_iter){
                bool is_left = x->isLeft(); // This gonna check if the current node is left or not
                x = x->parent; // Always getting the parent of the parent
                tree_iter = x->isRoot ? false : true; // To stop the while
                if(x->var_split == var_split){
                        tree_iter = false ; // This stop is necessary otherwise we would go up til the root, since we are always update there is no prob.
                        if(is_left){
                                upper = x->var_split_rule;
                                lower = x->lower;
                        } else {
                                upper = x->upper;
                                lower = x->var_split_rule;
                        }
                }
        }
}


void Node::displayCurrNode(){

                std::cout << "Node address: " << this << std::endl;
                std::cout << "Node parent: " << parent << std::endl;

                std::cout << "Cur Node is leaf: " << isLeaf << std::endl;
                std::cout << "Cur Node is root: " << isRoot << std::endl;
                std::cout << "Cur The split_var is: " << var_split << std::endl;
                std::cout << "Cur The split_var_rule is: " << var_split_rule << std::endl;

                return;
}


void Node::deletingLeaves(){

     // Should I create some warn to avoid memoery leak
     //something like it will only delete from a nog?
     // Deleting
     delete left; // This release the memory from the left point
     delete right; // This release the memory from the right point
     left = this;  // The new pointer for the left become the node itself
     right = this; // The new pointer for the right become the node itself
     isLeaf = true;

     return;

}
// Getting the leaves (this is the function that gonna do the recursion the
//                      function below is the one that gonna initialise it)
void get_leaves(Node* x,  std::vector<Node*> &leaves_vec) {

        if(x->isLeaf){
                leaves_vec.push_back(x);
        } else {
                get_leaves(x->left, leaves_vec);
                get_leaves(x->right,leaves_vec);
        }

        return;

}



// Initialising a vector of nodes in a standard way
std::vector<Node*> leaves(Node* x) {
        std::vector<Node*> leaves_init(0); // Initialising a vector of a vector of pointers of nodes of size zero
        get_leaves(x,leaves_init);
        return(leaves_init);
}

// Sweeping the trees looking for nogs
void get_nogs(std::vector<Node*>& nogs, Node* node){
        if(!node->isLeaf){
                bool bool_left_is_leaf = node->left->isLeaf;
                bool bool_right_is_leaf = node->right->isLeaf;

                // Checking if the current one is a NOGs
                if(bool_left_is_leaf && bool_right_is_leaf){
                        nogs.push_back(node);
                } else { // Keep looking for other NOGs
                        get_nogs(nogs, node->left);
                        get_nogs(nogs, node->right);
                }
        }
}

// Creating the vectors of nogs
std::vector<Node*> nogs(Node* tree){
        std::vector<Node*> nogs_init(0);
        get_nogs(nogs_init,tree);
        return nogs_init;
}



// Initializing the forest
Forest::Forest(modelParam& data){

        // Creatina vector of size of number of trees
        trees.resize(data.n_tree);
        for(int  i=0;i<data.n_tree;i++){
                // Creating the stump for each tree
                trees[i] = new Node(data);
                // Filling up each stump for each tree
                trees[i]->Stump(data);
        }
}

// Function to delete one tree
// Forest::~Forest(){
//         for(int  i=0;i<trees.size();i++){
//                 delete trees[i];
//         }
// }

// Selecting a random node
Node* sample_node(std::vector<Node*> leaves_){

        // Getting the number of leaves
        int n_leaves = leaves_.size();
        // return(leaves_[std::rand()%n_leaves]);
        if((n_leaves == 0) || (n_leaves==1) ){
             return leaves_[0];
        } else {
             return(leaves_[arma::randi(arma::distr_param(0,(n_leaves-1)))]);
        }

}

// Grow a tree for a given rule
void grow_c(Node* tree, modelParam &data, arma::vec &curr_res_r, arma::vec hat_q){

        // Getting the number of terminal nodes
        std::vector<Node*> t_nodes = leaves(tree) ;
        std::vector<Node*> nog_nodes = nogs(tree);

        // Selecting one node to be sampled
        Node* g_node = sample_node(t_nodes);

        // Store all old quantities that will be used or not
        double old_lower = g_node->lower;
        double old_upper = g_node->upper;
        int old_var_split = g_node->var_split;
        double old_var_split_rule = g_node->var_split_rule;

        // Calculating the whole likelihood fo the tree
        for(int i = 0; i < t_nodes.size(); i++){
                // cout << "Error gpNodeLogLike" << endl;
                t_nodes[i]->nodeUpdateResiduals_c(data, curr_res_r,hat_q);
        }

        // Adding the leaves
        g_node->addingLeaves(data);

        // Selecting the var
        g_node-> sampleSplitVar(data);
        // Updating the limits
        g_node->getLimits();


        // Selecting a rule
        g_node->var_split_rule = (g_node->upper-g_node->lower)*arma::randu(arma::distr_param(0.0,1.0))+g_node->lower;

        // Create an aux for the left and right index
        int train_left_counter = 0;
        int train_right_counter = 0;

        int test_left_counter = 0;
        int test_right_counter = 0;

        // Updating the left and the right nodes
        for(int i = 0;i<data.x_train.n_rows;i++){
                if(g_node -> train_index[i] == -1 ){
                        g_node->left->n_leaf = train_left_counter;
                        g_node->right->n_leaf = train_right_counter;
                        break;
                }
                if(data.x_train(g_node->train_index[i],g_node->var_split)<g_node->var_split_rule){
                        g_node->left->train_index[train_left_counter] = g_node->train_index[i];
                        train_left_counter++;
                } else {
                        g_node->right->train_index[train_right_counter] = g_node->train_index[i];
                        train_right_counter++;
                }

        }


        // Updating the left and right nodes for the
        for(int i = 0;i<data.x_test.n_rows; i++){
                if(g_node -> test_index[i] == -1){
                        g_node->left->n_leaf_test = test_left_counter;
                        g_node->right->n_leaf_test = test_right_counter;
                        break;
                }
                if(data.x_test(g_node->test_index[i],g_node->var_split)<g_node->var_split_rule){
                        g_node->left->test_index[test_left_counter] = g_node->test_index[i];
                        test_left_counter++;
                } else {
                        g_node->right->test_index[test_right_counter] = g_node->test_index[i];
                        test_right_counter++;
                }
        }

        // If is a root node
        if(g_node->isRoot){
                g_node->left->n_leaf = train_left_counter;
                g_node->right->n_leaf = train_right_counter;
                g_node->left->n_leaf_test = test_left_counter;
                g_node->right->n_leaf_test = test_right_counter;
        }

        // Avoiding nodes lower than the node_min
        if((g_node->left->n_leaf<data.node_min_size) || (g_node->right->n_leaf<data.node_min_size) ){

                // cout << " NODES" << endl;
                // Returning to the old values
                g_node->var_split = old_var_split;
                g_node->var_split_rule = old_var_split_rule;
                g_node->lower = old_lower;
                g_node->upper = old_upper;
                g_node->deletingLeaves();
                return;
        }


        // Updating the loglikelihood for those terminal nodes
        // cout << "Error here" << endl;
        g_node->nodeLogLike_c(data,curr_res_r,hat_q);
        // cout << "MUST be here" << endl;

        // cout << "Calculating likelihood of the new node on left" << endl;
        g_node->left->nodeUpdateResiduals_c(data,curr_res_r,hat_q);
        g_node->left->nodeLogLike_c(data, curr_res_r,hat_q);
        // cout << "Calculating likelihood of the new node on right" << endl;
        g_node->right->nodeUpdateResiduals_c(data,curr_res_r,hat_q);
        g_node->right->nodeLogLike_c(data, curr_res_r,hat_q);


        // cout << "NodeLogLike ok again: " << g_node->left->sr_minus_sl << endl;


        // Calculating the prior term for the grow
        double tree_prior = log(data.alpha*pow((1+g_node->depth),-data.beta)) +
                log(1-data.alpha*pow((1+g_node->depth+1),-data.beta)) + // Prior of left node being terminal
                log(1-data.alpha*pow((1+g_node->depth+1),-data.beta)) - // Prior of the right noide being terminal
                log(1-data.alpha*pow((1+g_node->depth),-data.beta)); // Old current node being terminal

        // Getting the transition probability
        double log_transition_prob = log((0.3)/(nog_nodes.size()+1)) - log(0.3/t_nodes.size()); // 0.3 and 0.3 are the prob of Prune and Grow, respectively

        // Calculating the loglikelihood for the new branches
        double new_tree_log_like = - g_node->log_likelihood + g_node->left->log_likelihood + g_node->right->log_likelihood ;

        // Calculating the acceptance ratio
        double acceptance = exp(new_tree_log_like + log_transition_prob + tree_prior);


        // Keeping the new tree or not
        if(arma::randu(arma::distr_param(0.0,1.0)) < acceptance){
                // Do nothing just keep the new tree
                // cout << " ACCEPTED" << endl;
                data.move_acceptance(0)++;
        } else {
                // Returning to the old values
                g_node->var_split = old_var_split;
                g_node->var_split_rule = old_var_split_rule;
                g_node->lower = old_lower;
                g_node->upper = old_upper;
                g_node->deletingLeaves();
        }

        return;

}




// Grow a tree for a given rule
void grow_q(Node* tree, modelParam &data, arma::vec &curr_res_s, arma::vec & hat_c){

        // Getting the number of terminal nodes
        std::vector<Node*> t_nodes = leaves(tree) ;
        std::vector<Node*> nog_nodes = nogs(tree);

        // Selecting one node to be sampled
        Node* g_node = sample_node(t_nodes);

        // Store all old quantities that will be used or not
        double old_lower = g_node->lower;
        double old_upper = g_node->upper;
        int old_var_split = g_node->var_split;
        double old_var_split_rule = g_node->var_split_rule;



        // Calculating the whole likelihood fo the tree
        for(int i = 0; i < t_nodes.size(); i++){
                // cout << "Error gpNodeLogLike" << endl;
                t_nodes[i]->nodeUpdateResiduals_q(data, curr_res_s,hat_c);
        }

        // cout << "LogLike Node ok Grow" << endl;

        // Adding the leaves
        g_node->addingLeaves(data);

        // Selecting the var
        g_node-> sampleSplitVar(data);
        // Updating the limits
        g_node->getLimits();

        // Selecting a rule
        g_node->var_split_rule = (g_node->upper-g_node->lower)*arma::randu(arma::distr_param(0.0,1.0))+g_node->lower;

        // Create an aux for the left and right index
        int train_left_counter = 0;
        int train_right_counter = 0;

        int test_left_counter = 0;
        int test_right_counter = 0;

        // Updating the left and the right nodes
        for(int i = 0;i<data.x_train.n_rows;i++){
                if(g_node -> train_index[i] == -1 ){
                        g_node->left->n_leaf = train_left_counter;
                        g_node->right->n_leaf = train_right_counter;
                        break;
                }
                if(data.x_train(g_node->train_index[i],g_node->var_split)<g_node->var_split_rule){
                        g_node->left->train_index[train_left_counter] = g_node->train_index[i];
                        train_left_counter++;
                } else {
                        g_node->right->train_index[train_right_counter] = g_node->train_index[i];
                        train_right_counter++;
                }

        }


        // Updating the left and right nodes for the
        for(int i = 0;i<data.x_test.n_rows; i++){
                if(g_node -> test_index[i] == -1){
                        g_node->left->n_leaf_test = test_left_counter;
                        g_node->right->n_leaf_test = test_right_counter;
                        break;
                }
                if(data.x_test(g_node->test_index[i],g_node->var_split)<g_node->var_split_rule){
                        g_node->left->test_index[test_left_counter] = g_node->test_index[i];
                        test_left_counter++;
                } else {
                        g_node->right->test_index[test_right_counter] = g_node->test_index[i];
                        test_right_counter++;
                }
        }

        // If is a root node
        if(g_node->isRoot){
                g_node->left->n_leaf = train_left_counter;
                g_node->right->n_leaf = train_right_counter;
                g_node->left->n_leaf_test = test_left_counter;
                g_node->right->n_leaf_test = test_right_counter;
        }

        // Avoiding nodes lower than the node_min
        if((g_node->left->n_leaf<data.node_min_size) || (g_node->right->n_leaf<data.node_min_size) ){

                // cout << " NODES" << endl;
                // Returning to the old values
                g_node->var_split = old_var_split;
                g_node->var_split_rule = old_var_split_rule;
                g_node->lower = old_lower;
                g_node->upper = old_upper;
                g_node->deletingLeaves();
                return;
        }


        // Updating the loglikelihood for those terminal nodes
        g_node->nodeLogLike_q(data,curr_res_s,hat_c);
        // cout << "Calculating likelihood of the new node on left" << endl;
        g_node->left->nodeUpdateResiduals_q(data,curr_res_s,hat_c);
        g_node->left->nodeLogLike_q(data, curr_res_s,hat_c);
        // cout << "Calculating likelihood of the new node on right" << endl;
        g_node->right->nodeUpdateResiduals_q(data,curr_res_s,hat_c);
        g_node->right->nodeLogLike_q(data, curr_res_s,hat_c);
        // cout << "NodeLogLike ok again" << endl;


        // Calculating the prior term for the grow
        double tree_prior = log(data.alpha*pow((1+g_node->depth),-data.beta)) +
                log(1-data.alpha*pow((1+g_node->depth+1),-data.beta)) + // Prior of left node being terminal
                log(1-data.alpha*pow((1+g_node->depth+1),-data.beta)) - // Prior of the right noide being terminal
                log(1-data.alpha*pow((1+g_node->depth),-data.beta)); // Old current node being terminal

        // Getting the transition probability
        double log_transition_prob = log((0.3)/(nog_nodes.size()+1)) - log(0.3/t_nodes.size()); // 0.3 and 0.3 are the prob of Prune and Grow, respectively

        // Calculating the loglikelihood for the new branches
        double new_tree_log_like =  - g_node->log_likelihood + g_node->left->log_likelihood + g_node->right->log_likelihood ;

        // Calculating the acceptance ratio
        double acceptance = exp(new_tree_log_like  + log_transition_prob + tree_prior);


        // Keeping the new tree or not
        if(arma::randu(arma::distr_param(0.0,1.0)) < acceptance){
                // Do nothing just keep the new tree
                // cout << " ACCEPTED" << endl;
                data.move_acceptance(0)++;
        } else {
                // Returning to the old values
                g_node->var_split = old_var_split;
                g_node->var_split_rule = old_var_split_rule;
                g_node->lower = old_lower;
                g_node->upper = old_upper;
                g_node->deletingLeaves();
        }

        return;

}

// Pruning a tree
void prune_c(Node* tree, modelParam&data, arma::vec &curr_res_r, arma::vec &hat_q){


        // Getting the number of terminal nodes
        std::vector<Node*> t_nodes = leaves(tree);

        // Can't prune a root
        if(t_nodes.size()==1){
                // cout << "Nodes size " << t_nodes.size() <<endl;
                t_nodes[0]->nodeUpdateResiduals_c(data, curr_res_r,hat_q);
                t_nodes[0]->nodeLogLike_c(data,curr_res_r,hat_q);
                return;
        }

        std::vector<Node*> nog_nodes = nogs(tree);

        // Selecting one node to be sampled
        Node* p_node = sample_node(nog_nodes);

        // Calculating the whole likelihood fo the tree
        for(int i = 0; i < t_nodes.size(); i++){
                t_nodes[i]->nodeUpdateResiduals_c(data, curr_res_r,hat_q);
        }

        // Updating the loglikelihood of the selected pruned node and its children
        p_node->nodeUpdateResiduals_c(data,curr_res_r,hat_q);
        p_node->nodeLogLike_c(data, curr_res_r,hat_q);
        p_node->left->nodeLogLike_c(data,curr_res_r,hat_q);
        p_node->right->nodeLogLike_c(data,curr_res_r,hat_q);

        // Getting the loglikelihood of the new tree
        double new_tree_log_like =  p_node->log_likelihood - (p_node->left->log_likelihood + p_node->right->log_likelihood);

        // Calculating the transition loglikelihood
        double transition_loglike = log((0.3)/(t_nodes.size())) - log((0.3)/(nog_nodes.size()));

        // Calculating the prior term for the grow
        double tree_prior = log(1-data.alpha*pow((1+p_node->depth),-data.beta))-
                log(data.alpha*pow((1+p_node->depth),-data.beta)) -
                log(1-data.alpha*pow((1+p_node->depth+1),-data.beta)) - // Prior of left node being terminal
                log(1-data.alpha*pow((1+p_node->depth+1),-data.beta));  // Prior of the right noide being terminal
                 // Old current node being terminal


        // Calculating the acceptance
        double acceptance = exp(new_tree_log_like + transition_loglike + tree_prior);

        if(arma::randu(arma::distr_param(0.0,1.0))<acceptance){
                p_node->deletingLeaves();
                data.move_acceptance(1)++;
        } else {
                // p_node->left->gpNodeLogLike(data, curr_res);
                // p_node->right->gpNodeLogLike(data, curr_res);
        }

        return;
}


// Pruning a tree
void prune_q(Node* tree, modelParam&data, arma::vec &curr_res_s, arma::vec & hat_c){


        // Getting the number of terminal nodes
        std::vector<Node*> t_nodes = leaves(tree);

        // Can't prune a root
        if(t_nodes.size()==1){
                // cout << "Nodes size " << t_nodes.size() <<endl;
                t_nodes[0]->nodeUpdateResiduals_q(data, curr_res_s, hat_c);
                t_nodes[0]->nodeLogLike_q(data,curr_res_s,hat_c);
                return;
        }

        std::vector<Node*> nog_nodes = nogs(tree);

        // Selecting one node to be sampled
        Node* p_node = sample_node(nog_nodes);


        // Calculating the whole likelihood fo the tree
        for(int i = 0; i < t_nodes.size(); i++){
                t_nodes[i]->nodeUpdateResiduals_q(data, curr_res_s,hat_c);
        }

        // cout << "Error C1" << endl;
        // Updating the loglikelihood of the selected pruned node and its children
        p_node->nodeUpdateResiduals_q(data,curr_res_s,hat_c);
        p_node->nodeLogLike_q(data, curr_res_s,hat_c);
        p_node->left->nodeLogLike_q(data,curr_res_s,hat_c);
        p_node->right->nodeUpdateResiduals_q(data,curr_res_s,hat_c);
        // cout << "Error C2" << endl;

        // Getting the loglikelihood of the new tree
        double new_tree_log_like =  p_node->log_likelihood - (p_node->left->log_likelihood + p_node->right->log_likelihood);

        // Calculating the transition loglikelihood
        double transition_loglike = log((0.3)/(t_nodes.size())) - log((0.3)/(nog_nodes.size()));

        // Calculating the prior term for the grow
        double tree_prior = log(1-data.alpha*pow((1+p_node->depth),-data.beta))-
                log(data.alpha*pow((1+p_node->depth),-data.beta)) -
                log(1-data.alpha*pow((1+p_node->depth+1),-data.beta)) - // Prior of left node being terminal
                log(1-data.alpha*pow((1+p_node->depth+1),-data.beta));  // Prior of the right noide being terminal
        // Old current node being terminal


        // Calculating the acceptance
        double acceptance = exp(new_tree_log_like  + transition_loglike + tree_prior);

        if(arma::randu(arma::distr_param(0.0,1.0))<acceptance){
                p_node->deletingLeaves();
                data.move_acceptance(1)++;
        } else {
                // p_node->left->gpNodeLogLike(data, curr_res);
                // p_node->right->gpNodeLogLike(data, curr_res);
        }

        return;
}



// // Creating the change verb
void change_c(Node* tree, modelParam &data, arma::vec &curr_res_r,arma::vec &hat_q){


        // Getting the number of terminal nodes
        std::vector<Node*> t_nodes = leaves(tree) ;
        std::vector<Node*> nog_nodes = nogs(tree);

        // Selecting one node to be sampled
        Node* c_node = sample_node(nog_nodes);


        if(c_node->isRoot){
                // cout << " THAT NEVER HAPPENS" << endl;
               c_node-> n_leaf = data.x_train.n_rows;
               c_node-> n_leaf_test = data.x_test.n_rows;
        }

        // cout << " Change error on terminal nodes" << endl;
        // Calculating the whole likelihood fo the tree
        for(int i = 0; i < t_nodes.size(); i++){
                // cout << "Loglike error " << endl;
                t_nodes[i]->nodeUpdateResiduals_c(data, curr_res_r,hat_q);
        }
        // cout << " Other kind of error" << endl;
        // If the current node has size zero there is no point of change its rule
        if(c_node->n_leaf==0) {
                return;
        }

        // Calculating the loglikelihood from the left and right node
        // cout << "Loglike error left " << endl;

        c_node->left->nodeLogLike_c(data,curr_res_r,hat_q);
        // cout << "Loglike error right" << endl;

        c_node->right->nodeLogLike_c(data,curr_res_r,hat_q);

        // cout << "Node loglike are fine" << endl;


        // Storing all the old loglikelihood from left
        double old_left_log_like = c_node->left->log_likelihood;
        double old_left_sr_minus_sl = c_node->left->sr_minus_sl;
        double old_left_sr_minus_sl_sq = c_node->left->sr_minus_sl_sq;
        double old_left_s_r_minus_l_sq = c_node->left->s_r_minus_l_sq;
        double old_left_ss_minus_sm = c_node->left->ss_minus_sm;
        double old_left_ss_minus_sm_sq = c_node->left->ss_minus_sm_sq;
        double old_left_s_s_minus_m_sq = c_node->left->s_s_minus_m_sq;
        double old_left_gamma = c_node->left->gamma;
        double old_left_eta = c_node->left->eta;

        arma::vec old_left_train_index = c_node->left->train_index;
        c_node->left->train_index.fill(-1); // Returning to the original
        int old_left_n_leaf = c_node->left->n_leaf;


        // Storing all of the old loglikelihood from right;
        double old_right_log_like = c_node->right->log_likelihood;
        double old_right_sr_minus_sl = c_node->right->sr_minus_sl;
        double old_right_sr_minus_sl_sq = c_node->right->sr_minus_sl_sq;
        double old_right_s_r_minus_l_sq = c_node->right->s_r_minus_l_sq;
        double old_right_ss_minus_sm = c_node->right->ss_minus_sm;
        double old_right_ss_minus_sm_sq = c_node->right->ss_minus_sm_sq;
        double old_right_s_s_minus_m_sq = c_node->right->s_s_minus_m_sq;
        double old_right_gamma = c_node->right->gamma;
        double old_right_eta = c_node->right->eta;

        arma::vec old_right_train_index = c_node->right->train_index;
        c_node->right->train_index.fill(-1);
        int old_right_n_leaf = c_node->right->n_leaf;



        // Storing test observations
        arma::vec old_left_test_index = c_node->left->test_index;
        arma::vec old_right_test_index = c_node->right->test_index;
        c_node->left->test_index.fill(-1);
        c_node->right->test_index.fill(-1);

        int old_left_n_leaf_test = c_node->left->n_leaf_test;
        int old_right_n_leaf_test = c_node->right->n_leaf_test;


        // Storing the old ones
        int old_var_split = c_node->var_split;
        int old_var_split_rule = c_node->var_split_rule;
        int old_lower = c_node->lower;
        int old_upper = c_node->upper;

        // Selecting the var
        c_node-> sampleSplitVar(data);
        // Updating the limits
        c_node->getLimits();
        // Selecting a rule
        c_node -> var_split_rule = (c_node->upper-c_node->lower)*arma::randu(arma::distr_param(0.0,1.0))+c_node->lower;
        // c_node -> var_split_rule = 0.0;

        // Create an aux for the left and right index
        int train_left_counter = 0;
        int train_right_counter = 0;

        int test_left_counter = 0;
        int test_right_counter = 0;



        // Updating the left and the right nodes
        for(int i = 0;i<data.x_train.n_rows;i++){
                // cout << " Train indexeses " << c_node -> train_index[i] << endl ;
                if(c_node -> train_index[i] == -1){
                        c_node->left->n_leaf = train_left_counter;
                        c_node->right->n_leaf = train_right_counter;
                        break;
                }
                // cout << " Current train index " << c_node->train_index[i] << endl;

                if(data.x_train(c_node->train_index[i],c_node->var_split)<c_node->var_split_rule){
                        c_node->left->train_index[train_left_counter] = c_node->train_index[i];
                        train_left_counter++;
                } else {
                        c_node->right->train_index[train_right_counter] = c_node->train_index[i];
                        train_right_counter++;
                }
        }



        // Updating the left and the right nodes
        for(int i = 0;i<data.x_test.n_rows;i++){

                if(c_node -> test_index[i] == -1){
                        c_node->left->n_leaf_test = test_left_counter;
                        c_node->right->n_leaf_test = test_right_counter;
                        break;
                }

                if(data.x_test(c_node->test_index[i],c_node->var_split)<c_node->var_split_rule){
                        c_node->left->test_index[test_left_counter] = c_node->test_index[i];
                        test_left_counter++;
                } else {
                        c_node->right->test_index[test_right_counter] = c_node->test_index[i];
                        test_right_counter++;
                }
        }

        // If is a root node
        if(c_node->isRoot){
                c_node->left->n_leaf = train_left_counter;
                c_node->right->n_leaf = train_right_counter;
                c_node->left->n_leaf_test = test_left_counter;
                c_node->right->n_leaf_test = test_right_counter;
        }


        if((c_node->left->n_leaf<data.node_min_size) || (c_node->right->n_leaf)<data.node_min_size){

                // Returning to the previous values
                c_node->var_split = old_var_split;
                c_node->var_split_rule = old_var_split_rule;
                c_node->lower = old_lower;
                c_node->upper = old_upper;

                // Returning to the old ones
                c_node->left->sr_minus_sl = old_left_sr_minus_sl;
                c_node->left->sr_minus_sl_sq = old_left_sr_minus_sl_sq;
                c_node->left->s_r_minus_l_sq = old_left_s_r_minus_l_sq;
                c_node->left->ss_minus_sm = old_left_ss_minus_sm;
                c_node->left->ss_minus_sm_sq = old_left_ss_minus_sm_sq;
                c_node->left->s_s_minus_m_sq = old_left_s_s_minus_m_sq;
                c_node->left->gamma = old_left_gamma;
                c_node->left->eta = old_left_eta;



                c_node->left->n_leaf = old_left_n_leaf;
                c_node->left->n_leaf_test = old_left_n_leaf_test;
                c_node->left->log_likelihood = old_left_log_like;
                c_node->left->train_index = old_left_train_index;
                c_node->left->test_index = old_left_test_index;

                // Returning to the old ones
                c_node->right->sr_minus_sl = old_right_sr_minus_sl;
                c_node->right->sr_minus_sl_sq = old_right_sr_minus_sl_sq;
                c_node->right->s_r_minus_l_sq = old_right_s_r_minus_l_sq;
                c_node->right->ss_minus_sm = old_right_ss_minus_sm;
                c_node->right->ss_minus_sm_sq = old_right_ss_minus_sm_sq;
                c_node->right->s_s_minus_m_sq = old_right_s_s_minus_m_sq;
                c_node->right->gamma = old_right_gamma;
                c_node->right->eta = old_right_eta;


                c_node->right->n_leaf = old_right_n_leaf;
                c_node->right->n_leaf_test = old_right_n_leaf_test;
                c_node->right->log_likelihood = old_right_log_like;
                c_node->right->train_index = old_right_train_index;
                c_node->right->test_index = old_right_test_index;

                return;
        }

        // Updating the new left and right loglikelihoods (need to update the residuals as well)
        // cout << " (NEW)Loglike error " << endl;

        c_node->left->nodeUpdateResiduals_c(data,curr_res_r,hat_q);
        c_node->left->nodeLogLike_c(data,curr_res_r,hat_q);
        c_node->right->nodeUpdateResiduals_c(data,curr_res_r,hat_q);
        c_node->right->nodeLogLike_c(data,curr_res_r,hat_q);

        // cout << " END Loglike error " << endl;

        // Calculating the acceptance
        double new_tree_log_like =  - old_left_log_like - old_right_log_like + c_node->left->log_likelihood + c_node->right->log_likelihood;

        double acceptance = exp(new_tree_log_like);

        if(arma::randu(arma::distr_param(0.0,1.0))<acceptance){
                // Keep all the trees
                data.move_acceptance(2)++;
        } else {

                // Returning to the previous values
                c_node->var_split = old_var_split;
                c_node->var_split_rule = old_var_split_rule;
                c_node->lower = old_lower;
                c_node->upper = old_upper;

                // Returning to the old ones
                c_node->left->sr_minus_sl = old_left_sr_minus_sl;
                c_node->left->sr_minus_sl_sq = old_left_sr_minus_sl_sq;
                c_node->left->s_r_minus_l_sq = old_left_s_r_minus_l_sq;
                c_node->left->ss_minus_sm = old_left_ss_minus_sm;
                c_node->left->ss_minus_sm_sq = old_left_ss_minus_sm_sq;
                c_node->left->s_s_minus_m_sq = old_left_s_s_minus_m_sq;
                c_node->left->gamma = old_left_gamma;
                c_node->left->eta = old_left_eta;

                c_node->left->n_leaf = old_left_n_leaf;
                c_node->left->n_leaf_test = old_left_n_leaf_test;
                c_node->left->log_likelihood = old_left_log_like;
                c_node->left->train_index = old_left_train_index;
                c_node->left->test_index = old_left_test_index;

                // Returning to the old ones
                c_node->right->sr_minus_sl = old_right_sr_minus_sl;
                c_node->right->sr_minus_sl_sq = old_right_sr_minus_sl_sq;
                c_node->right->s_r_minus_l_sq = old_right_s_r_minus_l_sq;
                c_node->right->ss_minus_sm = old_right_ss_minus_sm;
                c_node->right->ss_minus_sm_sq = old_right_ss_minus_sm_sq;
                c_node->right->s_s_minus_m_sq = old_right_s_s_minus_m_sq;
                c_node->right->gamma = old_right_gamma;
                c_node->right->eta = old_right_eta;

                c_node->right->n_leaf = old_right_n_leaf;
                c_node->right->n_leaf_test = old_right_n_leaf_test;
                c_node->right->log_likelihood = old_right_log_like;
                c_node->right->train_index = old_right_train_index;
                c_node->right->test_index = old_right_test_index;

        }

        return;
}


// // Creating the change verb
void change_q(Node* tree, modelParam &data, arma::vec &curr_res_s,arma::vec &hat_c){


        // Getting the number of terminal nodes
        std::vector<Node*> t_nodes = leaves(tree) ;
        std::vector<Node*> nog_nodes = nogs(tree);

        // Selecting one node to be sampled
        Node* c_node = sample_node(nog_nodes);

        if(c_node->isRoot){
                // cout << " THAT NEVER HAPPENS" << endl;
                c_node-> n_leaf = data.x_train.n_rows;
                c_node-> n_leaf_test = data.x_test.n_rows;
        }

        // cout << " Change error on terminal nodes" << endl;
        // Calculating the whole likelihood fo the tree
        for(int i = 0; i < t_nodes.size(); i++){
                // cout << "Loglike error " << ed
                t_nodes[i]->nodeUpdateResiduals_q(data, curr_res_s,hat_c);
        }
        // cout << " Other kind of error" << endl;
        // If the current node has size zero there is no point of change its rule
        if(c_node->n_leaf==0) {
                return;
        }

        // Calculating the loglikelihood from the left and right node
        c_node->left->nodeLogLike_q(data,curr_res_s,hat_c);
        c_node->right->nodeLogLike_q(data,curr_res_s,hat_c);

        // Storing all the old loglikelihood from left
        double old_left_log_like = c_node->left->log_likelihood;
        double old_left_sr_minus_sl = c_node->left->sr_minus_sl;
        double old_left_sr_minus_sl_sq = c_node->left->sr_minus_sl_sq;
        double old_left_s_r_minus_l_sq = c_node->left->s_r_minus_l_sq;
        double old_left_ss_minus_sm = c_node->left->ss_minus_sm;
        double old_left_ss_minus_sm_sq = c_node->left->ss_minus_sm_sq;
        double old_left_s_s_minus_m_sq = c_node->left->s_s_minus_m_sq;
        double old_left_gamma = c_node->left->gamma;
        double old_left_eta = c_node->left->eta;


        arma::vec old_left_train_index = c_node->left->train_index;
        c_node->left->train_index.fill(-1); // Returning to the original
        int old_left_n_leaf = c_node->left->n_leaf;


        // Storing all of the old loglikelihood from right;
        double old_right_log_like = c_node->right->log_likelihood;
        double old_right_sr_minus_sl = c_node->right->sr_minus_sl;
        double old_right_sr_minus_sl_sq = c_node->right->sr_minus_sl_sq;
        double old_right_s_r_minus_l_sq = c_node->right->s_r_minus_l_sq;
        double old_right_ss_minus_sm = c_node->right->ss_minus_sm;
        double old_right_ss_minus_sm_sq = c_node->right->ss_minus_sm_sq;
        double old_right_s_s_minus_m_sq = c_node->right->s_s_minus_m_sq;
        double old_right_gamma = c_node->right->gamma;
        double old_right_eta = c_node->right->eta;

        arma::vec old_right_train_index = c_node->right->train_index;
        c_node->right->train_index.fill(-1);
        int old_right_n_leaf = c_node->right->n_leaf;



        // Storing test observations
        arma::vec old_left_test_index = c_node->left->test_index;
        arma::vec old_right_test_index = c_node->right->test_index;
        c_node->left->test_index.fill(-1);
        c_node->right->test_index.fill(-1);

        int old_left_n_leaf_test = c_node->left->n_leaf_test;
        int old_right_n_leaf_test = c_node->right->n_leaf_test;


        // Storing the old ones
        int old_var_split = c_node->var_split;
        int old_var_split_rule = c_node->var_split_rule;
        int old_lower = c_node->lower;
        int old_upper = c_node->upper;

        // Selecting the var
        c_node-> sampleSplitVar(data);
        // Updating the limits
        c_node->getLimits();
        // Selecting a rule
        c_node -> var_split_rule = (c_node->upper-c_node->lower)*arma::randu(arma::distr_param(0.0,1.0))+c_node->lower;
        // c_node -> var_split_rule = 0.0;

        // Create an aux for the left and right index
        int train_left_counter = 0;
        int train_right_counter = 0;

        int test_left_counter = 0;
        int test_right_counter = 0;


        // Updating the left and the right nodes
        for(int i = 0;i<data.x_train.n_rows;i++){
                // cout << " Train indexeses " << c_node -> train_index[i] << endl ;
                if(c_node -> train_index[i] == -1){
                        c_node->left->n_leaf = train_left_counter;
                        c_node->right->n_leaf = train_right_counter;
                        break;
                }
                // cout << " Current train index " << c_node->train_index[i] << endl;

                if(data.x_train(c_node->train_index[i],c_node->var_split)<c_node->var_split_rule){
                        c_node->left->train_index[train_left_counter] = c_node->train_index[i];
                        train_left_counter++;
                } else {
                        c_node->right->train_index[train_right_counter] = c_node->train_index[i];
                        train_right_counter++;
                }
        }



        // Updating the left and the right nodes
        for(int i = 0;i<data.x_test.n_rows;i++){

                if(c_node -> test_index[i] == -1){
                        c_node->left->n_leaf_test = test_left_counter;
                        c_node->right->n_leaf_test = test_right_counter;
                        break;
                }

                if(data.x_test(c_node->test_index[i],c_node->var_split)<c_node->var_split_rule){
                        c_node->left->test_index[test_left_counter] = c_node->test_index[i];
                        test_left_counter++;
                } else {
                        c_node->right->test_index[test_right_counter] = c_node->test_index[i];
                        test_right_counter++;
                }
        }

        // If is a root node
        if(c_node->isRoot){
                c_node->left->n_leaf = train_left_counter;
                c_node->right->n_leaf = train_right_counter;
                c_node->left->n_leaf_test = test_left_counter;
                c_node->right->n_leaf_test = test_right_counter;
        }


        if((c_node->left->n_leaf<data.node_min_size) || (c_node->right->n_leaf)<data.node_min_size){

                // Returning to the previous values
                c_node->var_split = old_var_split;
                c_node->var_split_rule = old_var_split_rule;
                c_node->lower = old_lower;
                c_node->upper = old_upper;

                // Returning to the old ones
                c_node->left->sr_minus_sl = old_left_sr_minus_sl;
                c_node->left->sr_minus_sl_sq = old_left_sr_minus_sl_sq;
                c_node->left->s_r_minus_l_sq = old_left_s_r_minus_l_sq;
                c_node->left->ss_minus_sm = old_left_ss_minus_sm;
                c_node->left->ss_minus_sm_sq = old_left_ss_minus_sm_sq;
                c_node->left->s_s_minus_m_sq = old_left_s_s_minus_m_sq;
                c_node->left->gamma = old_left_gamma;
                c_node->left->eta = old_left_eta;

                c_node->left->n_leaf = old_left_n_leaf;
                c_node->left->n_leaf_test = old_left_n_leaf_test;
                c_node->left->log_likelihood = old_left_log_like;
                c_node->left->train_index = old_left_train_index;
                c_node->left->test_index = old_left_test_index;

                // Returning to the old ones
                c_node->right->sr_minus_sl = old_right_sr_minus_sl;
                c_node->right->sr_minus_sl_sq = old_right_sr_minus_sl_sq;
                c_node->right->s_r_minus_l_sq = old_right_s_r_minus_l_sq;
                c_node->right->ss_minus_sm = old_right_ss_minus_sm;
                c_node->right->ss_minus_sm_sq = old_right_ss_minus_sm_sq;
                c_node->right->s_s_minus_m_sq = old_right_s_s_minus_m_sq;
                c_node->right->gamma = old_right_gamma;
                c_node->right->eta = old_right_eta;

                c_node->right->n_leaf = old_right_n_leaf;
                c_node->right->n_leaf_test = old_right_n_leaf_test;
                c_node->right->log_likelihood = old_right_log_like;
                c_node->right->train_index = old_right_train_index;
                c_node->right->test_index = old_right_test_index;

                return;
        }

        // Updating the new left and right loglikelihoods (need to update the residuals as well)
        c_node->left->nodeUpdateResiduals_q(data,curr_res_s,hat_c);
        c_node->left->nodeLogLike_q(data,curr_res_s,hat_c);
        c_node->right->nodeUpdateResiduals_q(data,curr_res_s,hat_c);
        c_node->right->nodeLogLike_q(data,curr_res_s,hat_c);

        // Calculating the acceptance
        double new_tree_log_like =  - old_left_log_like - old_right_log_like + c_node->left->log_likelihood + c_node->right->log_likelihood;

        double acceptance = exp(new_tree_log_like);

        if(arma::randu(arma::distr_param(0.0,1.0))<acceptance){
                // Keep all the trees
                data.move_acceptance(2)++;
        } else {

                // Returning to the previous values
                c_node->var_split = old_var_split;
                c_node->var_split_rule = old_var_split_rule;
                c_node->lower = old_lower;
                c_node->upper = old_upper;

                // Returning to the old ones
                c_node->left->sr_minus_sl = old_left_sr_minus_sl;
                c_node->left->sr_minus_sl_sq = old_left_sr_minus_sl_sq;
                c_node->left->s_r_minus_l_sq = old_left_s_r_minus_l_sq;
                c_node->left->ss_minus_sm = old_left_ss_minus_sm;
                c_node->left->ss_minus_sm_sq = old_left_ss_minus_sm_sq;
                c_node->left->s_s_minus_m_sq = old_left_s_s_minus_m_sq;
                c_node->left->gamma = old_left_gamma;
                c_node->left->eta = old_left_eta;

                c_node->left->n_leaf = old_left_n_leaf;
                c_node->left->n_leaf_test = old_left_n_leaf_test;
                c_node->left->log_likelihood = old_left_log_like;
                c_node->left->train_index = old_left_train_index;
                c_node->left->test_index = old_left_test_index;

                // Returning to the old ones
                c_node->right->sr_minus_sl = old_right_sr_minus_sl;
                c_node->right->sr_minus_sl_sq = old_right_sr_minus_sl_sq;
                c_node->right->s_r_minus_l_sq = old_right_s_r_minus_l_sq;
                c_node->right->ss_minus_sm = old_right_ss_minus_sm;
                c_node->right->ss_minus_sm_sq = old_right_ss_minus_sm_sq;
                c_node->right->s_s_minus_m_sq = old_right_s_s_minus_m_sq;
                c_node->right->gamma = old_right_gamma;
                c_node->right->eta = old_right_eta;


                c_node->right->n_leaf = old_right_n_leaf;
                c_node->right->n_leaf_test = old_right_n_leaf_test;
                c_node->right->log_likelihood = old_right_log_like;
                c_node->right->train_index = old_right_train_index;
                c_node->right->test_index = old_right_test_index;

        }

        return;
}



// Calculating the Loglikelihood of a node
void Node::nodeUpdateResiduals_c(modelParam& data,
                                arma::vec &curr_res_r,
                                arma::vec &hat_q){

        // Getting number of leaves in case of a root
        if(isRoot){
                // Updating the r_sum
                n_leaf = data.x_train.n_rows;
                n_leaf_test = data.x_test.n_rows;
        }


        // Case of an empty node
        if(train_index[0]==-1){
                // if(n_leaf < 100){
                n_leaf = 0;
                /// Initialising the residuals sum
                sr_minus_sl = 0.0;
                sr_minus_sl_sq = 0.0;
                s_r_minus_l_sq = 0.0;
                Rcpp::stop("Error");
                log_likelihood = -2000000; // Absurd value avoid this case
                // cout << "OOOPS something happened" << endl;
                return;
        }

        // If is smaller then the node size still need to update the quantities;
        // cout << "Node min size: " << data.node_min_size << endl;
        if(n_leaf < data.node_min_size){
                /// Initialising the residuals sum
                sr_minus_sl = 0.0;
                sr_minus_sl_sq = 0.0;
                s_r_minus_l_sq = 0.0;
                Rcpp::stop("Error");

                log_likelihood = -2000000; // Absurd value avoid this case
                // cout << "OOOPS something happened" << endl;
                return;
        }

        /// Initialising the residuals sum
        sr_minus_sl = 0.0;
        sr_minus_sl_sq = 0.0;
        s_r_minus_l_sq = 0.0;


        // Creating the vector that corresponds to the l vector
        // cout << "Q(train): " << data.q_train(0) << endl;
        // cout << "Haaat --- Q(train): " << hat_q(0) << endl;
        // cout << "Pho: " << data.rho << endl;

        arma::vec l = (sqrt(data.tau_q)/sqrt(data.tau_c))*data.rho*(data.q_train-hat_q);
        arma::vec leaf_r(n_leaf,arma::fill::zeros);
        arma::vec leaf_l(n_leaf,arma::fill::zeros);

        if(n_leaf==0){
                Rcpp::stop("Error empty bide");
        }

        // Train elements
        for(int i = 0; i < n_leaf;i++){
                leaf_l(i) = l(train_index[i]);
                leaf_r(i) = curr_res_r(train_index[i]);
                s_r_minus_l_sq = s_r_minus_l_sq + (leaf_r(i)-leaf_l(i))*(leaf_r(i)-leaf_l(i));
        }

        // cout << "Sum of l_i" << arma::sum(leaf_l) << endl;
        // cout << "Leaf size: " << n_leaf << endl;
        // cout << train_index << endl;
        // cout << "Sum of residuals_i " << arma::sum(leaf_r)<< endl;


        sr_minus_sl = arma::sum(leaf_r)-arma::sum(leaf_l);
        sr_minus_sl_sq = sr_minus_sl*sr_minus_sl;


        // Updating the gamma
        gamma = n_leaf + (data.tau_mu*(1-data.rho*data.rho))/(data.tau_c);

        return;

}

// Calculating the Loglilelihood of a node
void Node::nodeLogLike_c(modelParam& data,
                        arma::vec &curr_res_r,
                        arma::vec &hat_q){

        // Getting number of leaves in case of a root
        if(isRoot){
                // Updating the r_sum
                n_leaf = data.x_train.n_rows;
                n_leaf_test = data.x_test.n_rows;
        }


        // Case of an empty node
        if(train_index[0]==-1){
        // if(n_leaf < 100){
                n_leaf = 0;
                log_likelihood = -2000000; // Absurd value avoid this case
                // cout << "OOOPS something happened" << endl;
                return;
        }

        // If is smaller then the node size still need to update the quantities;
        // cout << "Node min size: " << data.node_min_size << endl;
        if(n_leaf < data.node_min_size){
                log_likelihood = -2000000; // Absurd value avoid this case
                // cout << "OOOPS something happened" << endl;
                return;
        }

        // Getting the log-likelihood;
        log_likelihood = -0.5*n_leaf*log((1-data.rho*data.rho)/data.tau_c) + 0.5*log((1-data.rho*data.rho)/(data.tau_c*gamma)) -0.5*(data.tau_c/(1-data.rho*data.rho))*s_r_minus_l_sq - 0.5*((data.tau_c*gamma)/(1-data.rho*data.rho))*sr_minus_sl_sq;

        return;

}


// UPDATING MU ( NOT NECESSARY)
void updateMu(Node* tree, modelParam &data){

        // Getting the terminal nodes
        std::vector<Node*> t_nodes = leaves(tree);
        // Iterating over the terminal nodes and updating the beta values
        for(int i = 0; i < t_nodes.size();i++){
                // cout << "Analzying sum_r - sum_l: " <<  t_nodes[i]->sr_minus_sl << endl;
                // cout << "Gamma: " << t_nodes[i]->gamma << endl;
                // cout << "Getting the mean: " << (t_nodes[i]->sr_minus_sl) << endl;
                t_nodes[i]->mu = R::rnorm((t_nodes[i]->sr_minus_sl)/(t_nodes[i]->gamma),sqrt((1-data.rho*data.rho)/(data.tau_c*t_nodes[i]->gamma))) ;
                // cout << "SAMPLED MU: " << t_nodes[i]->mu << endl;
        }
}



// Calculating the Loglikelihood of a node
void Node::nodeUpdateResiduals_q(modelParam& data,
                                 arma::vec &curr_res_s,
                                 arma::vec &hat_c){

        // Getting number of leaves in case of a root
        if(isRoot){
                // Updating the r_sum
                n_leaf = data.x_train.n_rows;
                n_leaf_test = data.x_test.n_rows;
        }


        // Case of an empty node
        if(train_index[0]==-1){
                // if(n_leaf < 100){
                n_leaf = 0;
                /// Initialising the residuals sum
                ss_minus_sm = 0.0;
                ss_minus_sm_sq = 0.0;
                s_s_minus_m_sq = 0.0;
                log_likelihood = -2000000; // Absurd value avoid this case
                // cout << "OOOPS something happened" << endl;
                return;
        }

        // If is smaller then the node size still need to update the quantities;
        // cout << "Node min size: " << data.node_min_size << endl;
        if(n_leaf < data.node_min_size){
                /// Initialising the residuals sum
                ss_minus_sm = 0.0;
                ss_minus_sm_sq = 0.0;
                s_s_minus_m_sq = 0.0;
                log_likelihood = -2000000; // Absurd value avoid this case
                // cout << "OOOPS something happened" << endl;
                return;
        }

        /// Initialising the residuals sum
        ss_minus_sm = 0.0;
        ss_minus_sm_sq = 0.0;
        s_s_minus_m_sq = 0.0;

        // Creating the vector that corresponds to the l vector
        arma::vec m = (sqrt(data.tau_c)/sqrt(data.tau_q))*data.rho*(data.c_train-hat_c);
        arma::vec leaf_s(n_leaf);
        arma::vec leaf_m(n_leaf);

        // Train elements
        for(int i = 0; i < n_leaf;i++){
                leaf_m(i) = m(train_index[i]);
                leaf_s(i) = curr_res_s(train_index[i]);
                s_s_minus_m_sq = s_s_minus_m_sq + (leaf_s(i)-leaf_m(i))*(leaf_s(i)-leaf_m(i));
        }

        ss_minus_sm = arma::sum(leaf_s)-arma::sum(leaf_m);
        ss_minus_sm_sq = ss_minus_sm*ss_minus_sm;

        // Updating the eta
        eta = n_leaf + (data.tau_lambda*(1-data.rho*data.rho))/(data.tau_q);

        return;

}

// Calculating the Loglilelihood of a node
void Node::nodeLogLike_q(modelParam& data,
                         arma::vec &curr_res_s,
                         arma::vec &hat_q){

        // Getting number of leaves in case of a root
        if(isRoot){
                // Updating the r_sum
                n_leaf = data.x_train.n_rows;
                n_leaf_test = data.x_test.n_rows;
        }


        // Case of an empty node
        if(train_index[0]==-1){
                // if(n_leaf < 100){
                n_leaf = 0;
                log_likelihood = -2000000; // Absurd value avoid this case
                // cout << "OOOPS something happened" << endl;
                return;
        }

        // If is smaller then the node size still need to update the quantities;
        // cout << "Node min size: " << data.node_min_size << endl;
        if(n_leaf < data.node_min_size){
                log_likelihood = -2000000; // Absurd value avoid this case
                // cout << "OOOPS something happened" << endl;
                return;
        }

        // Getting the log-likelihood;
        log_likelihood = -0.5*n_leaf*log((1-data.rho*data.rho)/data.tau_q) + 0.5*log((1-data.rho*data.rho)/(data.tau_q*eta)) -0.5*(data.tau_q/(1-data.rho*data.rho))*s_s_minus_m_sq - 0.5*((data.tau_q*eta)/(1-data.rho*data.rho))*ss_minus_sm_sq;

        return;

}


// UPDATING LAMBDA
void updateLambda(Node* tree, modelParam &data){

        // Getting the terminal nodes
        std::vector<Node*> t_nodes = leaves(tree);

        // Iterating over the terminal nodes and updating the beta values
        for(int i = 0; i < t_nodes.size();i++){
                t_nodes[i]->lambda = R::rnorm((t_nodes[i]->ss_minus_sm)/(t_nodes[i]->eta),sqrt((1-data.rho*data.rho)/(data.tau_q*t_nodes[i]->eta))) ;
        }
}





// Get the prediction
// (MOST IMPORTANT AND COSTFUL FUNCTION FROM GP-BART)
void getPredictions_c(Node* tree,
                    modelParam &data,
                    arma::vec& current_prediction_train_c,
                    arma::vec& current_prediction_test_c){

        // Getting the current prediction
        vector<Node*> t_nodes = leaves(tree);
        for(int i = 0; i<t_nodes.size();i++){

                // Skipping empty nodes
                if(t_nodes[i]->n_leaf==0){
                        cout << " THERE ARE EMPTY NODES" << endl;
                        continue;
                }


                // For the training samples
                for(int j = 0; j<data.x_train.n_rows; j++){

                        if((t_nodes[i]->train_index[j])==-1){
                                break;
                        }
                        current_prediction_train_c[t_nodes[i]->train_index[j]] = t_nodes[i]->mu;

                }

                if(t_nodes[i]->n_leaf_test == 0 ){
                        continue;
                }



                // Regarding the test samples
                for(int j = 0; j< data.x_test.n_rows;j++){

                        if(t_nodes[i]->test_index[j]==-1){
                                break;
                        }

                        current_prediction_test_c[t_nodes[i]->test_index[j]] = t_nodes[i]->mu;
                }

        }
}

// Get the prediction
// (MOST IMPORTANT AND COSTFUL FUNCTION FROM GP-BART)
void getPredictions_q(Node* tree,
                      modelParam &data,
                      arma::vec& current_prediction_train_q,
                      arma::vec& current_prediction_test_q){

        // Getting the current prediction
        vector<Node*> t_nodes = leaves(tree);
        for(int i = 0; i<t_nodes.size();i++){

                // Skipping empty nodes
                if(t_nodes[i]->n_leaf==0){
                        cout << " THERE ARE EMPTY NODES" << endl;
                        continue;
                }


                // For the training samples
                for(int j = 0; j<data.x_train.n_rows; j++){

                        if((t_nodes[i]->train_index[j])==-1){
                                break;
                        }
                        current_prediction_train_q[t_nodes[i]->train_index[j]] = t_nodes[i]->lambda;

                }

                if(t_nodes[i]->n_leaf_test == 0 ){
                        continue;
                }



                // Regarding the test samples
                for(int j = 0; j< data.x_test.n_rows;j++){

                        if(t_nodes[i]->test_index[j]==-1){
                                break;
                        }

                        current_prediction_test_q[t_nodes[i]->test_index[j]] = t_nodes[i]->lambda;
                }

        }
}



// Updating the tau parameter
void updateP(arma::vec &c_hat,
             arma::vec &q_hat,
             modelParam &data){

        // Getting the sum of residuals square
        int n_ = c_hat.size();
        arma::mat S(data.s_0_wish.n_rows,data.s_0_wish.n_cols,arma::fill::zeros);
        arma::mat P_aux(data.s_0_wish.n_rows,data.s_0_wish.n_cols);

        for(int i = 0; i < data.x_train.n_rows; i ++ ){
                S(0,0) = S(0,0) + (data.c_train(i)-c_hat(i))*(data.c_train(i)-c_hat(i));
                double cov_aux = (data.c_train(i)-c_hat(i))*(data.q_train(i)-q_hat(i));
                S(0,1) = S(0,1) + cov_aux;
                S(1,0) = S(0,1) ;
                S(1,1) = S(1,1) + (data.q_train(i)-q_hat(i))*(data.q_train(i)-q_hat(i));
        }

        // cout << c_hat(0) << endl;
        // Replacing the values (it doesnt really matter the values for the inverse)
        P_aux = arma::inv(arma::wishrnd(arma::inv(S+data.s_0_wish),n_+data.df_wish));


        data.tau_c = 1/P_aux(0,0);
        data.tau_q = 1/P_aux(1,1);

        // cout << " Getting the covariance values" << P_aux(1,0) << endl;
        // cout << "Getting the covariance values" << P_aux(0,1) << endl;

        data.rho = P_aux(1,0)*(sqrt(data.tau_c)*sqrt(data.tau_q));

        return;
}


// Updating a new prior for S_0_wish
void update_s_0_wish(modelParam &data,
                     )



// Creating the BART function
// [[Rcpp::export]]
Rcpp::List cppbart(arma::mat x_train,
          arma::vec c_train,
          arma::vec q_train,
          arma::mat x_test,
          int n_tree,
          int node_min_size,
          double alpha,
          double beta,
          int n_mcmc,
          int n_burn,
          arma::mat P,
          double mu_c,
          double mu_q,
          double tau_mu,
          double tau_lambda,
          double df_wish,
          arma::mat s_0_wish){

        // Posterior counter
        int curr = 0;


        // cout << " Error on model.param" << endl;
        // Creating the structu object
        modelParam data(x_train,
                        c_train,
                        q_train,
                        x_test,
                        n_tree,
                        node_min_size,
                        alpha,
                        beta,
                        tau_mu,
                        tau_lambda,
                        df_wish,
                        s_0_wish,
                        n_mcmc,
                        n_burn);

        // Getting the n_post
        int n_post = n_mcmc - n_burn;


        // Replacing initial values for some of the parameters
        data.tau_c = (1/P(0,0));
        data.tau_q = (1/P(1,1));
        data.rho = P(1,0)*sqrt(data.tau_c)*sqrt(data.tau_q);


        // Defining those elements
        arma::mat c_train_hat_post = arma::zeros<arma::mat>(data.x_train.n_rows,n_post);
        arma::mat c_test_hat_post = arma::zeros<arma::mat>(data.x_test.n_rows,n_post);

        arma::mat q_train_hat_post = arma::zeros<arma::mat>(data.x_train.n_rows,n_post);
        arma::mat q_test_hat_post = arma::zeros<arma::mat>(data.x_test.n_rows,n_post);

        arma::vec tau_c_post = arma::zeros<arma::vec>(n_post);
        arma::vec tau_q_post = arma::zeros<arma::vec>(n_post);
        arma::vec rho_post = arma::zeros<arma::vec>(n_post);

        // Storing all parameters
        arma::vec all_tau_c(n_mcmc,arma::fill::zeros);
        arma::vec all_tau_q(n_mcmc,arma::fill::zeros);
        arma::vec all_rho(n_mcmc,arma::fill::zeros);

        // Defining other variables
        // arma::vec partial_pred_c(data.x_train.n_rows,arma::fill::zeros);
        // arma::vec partial_pred_q(data.x_train.n_rows,arma::fill::zeros);
        arma::vec partial_residuals_c = arma::zeros<arma::vec>(data.x_train.n_rows);
        arma::vec partial_residuals_q = arma::zeros<arma::vec>(data.x_train.n_rows);

        arma::mat tree_fits_store_c(data.x_train.n_rows,data.n_tree,arma::fill::zeros);
        arma::mat tree_fits_store_q(data.x_train.n_rows,data.n_tree,arma::fill::zeros);

        // Updating with small values
        // for(int i = 0 ; i < data.n_tree ; i ++ ){
        //         tree_fits_store_c.col(i) = partial_pred_c;
        //         tree_fits_store_q.col(i) = partial_pred_q;
        //
        // }
        arma::mat tree_fits_store_test_q(data.x_test.n_rows,data.n_tree,arma::fill::zeros);
        arma::mat tree_fits_store_test_c(data.x_test.n_rows,data.n_tree,arma::fill::zeros);

        double verb;

        // Defining progress bars parameters
        const int width = 70;
        double pb = 0;


        // cout << " Error one " << endl;

        // Selecting the train
        Forest all_forest_c(data);
        Forest all_forest_q(data);


        // Getting all the predictions sum
        arma::vec prediction_train_sum_c = arma::sum(tree_fits_store_c,1);
        arma::vec prediction_test_sum_c = arma::sum(tree_fits_store_test_c,1);

        arma::vec prediction_train_sum_q = arma::sum(tree_fits_store_q,1);
        arma::vec prediction_test_sum_q = arma::sum(tree_fits_store_test_q,1);

        for(int i = 0;i<data.n_mcmc;i++){

                // Initialising PB
                std::cout << "[";
                int k = 0;
                // Evaluating progress bar
                for(;k<=pb*width/data.n_mcmc;k++){
                        std::cout << "=";
                }

                for(; k < width;k++){
                        std:: cout << " ";
                }

                std::cout << "] " << std::setprecision(5) << (pb/data.n_mcmc)*100 << "%\r";
                std::cout.flush();

                // =============================
                // Iterating first over the cost;
                // =============================

                for(int t = 0; t<data.n_tree;t++){

                        // Creating the auxliar prediction vector
                        arma::vec c_hat(data.x_train.n_rows,arma::fill::zeros);
                        arma::vec prediction_test_c(data.x_test.n_rows,arma::fill::zeros);

                        // cout << "Residuals error "<< endl;
                        // Updating the partial residuals
                        if(data.n_tree>1){
                                partial_residuals_c = data.c_train-sum_exclude_col(tree_fits_store_c,t);
                        } else {
                                partial_residuals_c = data.c_train;
                        }


                        // cout << "Another error" << endl;
                        // Iterating over all trees
                        verb = arma::randu(arma::distr_param(0.0,1.0));
                        // cout << "Another error 2.0" << endl;

                        // Always growing
                        if(all_forest_c.trees[t]->isLeaf & all_forest_c.trees[t]->isRoot){
                                verb = arma::randu(arma::distr_param(0.0,0.3));
                        }

                        // Selecting the verb
                        if(verb < 0.3){
                                data.move_proposal(0)++;
                                // cout << " Grow error" << endl;
                                grow_c(all_forest_c.trees[t],data,partial_residuals_c,prediction_train_sum_q);

                        } else if(verb>=0.3 & verb <0.6) {
                                data.move_proposal(1)++;
                                // cout << " Prune error" << endl;
                                prune_c(all_forest_c.trees[t], data, partial_residuals_c,prediction_train_sum_q);
                        } else {
                                data.move_proposal(2)++;
                                // cout << " Change error C" << endl;
                                change_c(all_forest_c.trees[t], data, partial_residuals_c,prediction_train_sum_q);
                                // std::cout << "Error after change" << endl;
                        }

                        // std::cout << "Error on mu" << endl;
                        updateMu(all_forest_c.trees[t],data);

                        // Getting predictions
                        // cout << " Error on Get Predictions" << endl;
                        getPredictions_c(all_forest_c.trees[t],data,c_hat,prediction_test_c);

                        // Updating the tree
                        // cout << "Residuals error 2.0"<< endl;
                        tree_fits_store_c.col(t) = c_hat;
                        // cout << "Residuals error 3.0"<< endl;
                        tree_fits_store_test_c.col(t) = prediction_test_c;
                        // std::cout << "Residuals error 4.0"<< endl;

                }

                // Summing over all trees (cost trees)
                prediction_train_sum_c = sum(tree_fits_store_c,1);
                prediction_test_sum_c = sum(tree_fits_store_test_c,1);


                // =============================
                // Iterating secondly over the quality;
                // =============================

                for(int t = 0; t<data.n_tree;t++){

                        // Creating the auxliar prediction vector
                        arma::vec q_hat(data.x_train.n_rows,arma::fill::zeros);
                        arma::vec prediction_test_q(data.x_test.n_rows,arma::fill::zeros);

                        // cout << "Residuals error Q "<< endl;
                        // cout << " hat{Q}: "<< q_hat(0) << endl;

                        // Updating the partial residuals
                        if(data.n_tree>1){
                                partial_residuals_q = data.q_train-sum_exclude_col(tree_fits_store_q,t);
                        } else {
                                partial_residuals_q = data.q_train;
                        }

                        // Iterating over all trees
                        verb = arma::randu(arma::distr_param(0.0,1.0));

                        // Always growing
                        if(all_forest_q.trees[t]->isLeaf & all_forest_q.trees[t]->isRoot){
                                verb = arma::randu(arma::distr_param(0.0,0.3));
                        }

                        // Selecting the verb
                        if(verb < 0.3){
                                data.move_proposal(0)++;
                                // cout << " Grow error" << endl;
                                grow_q(all_forest_q.trees[t],data,partial_residuals_q,prediction_train_sum_c);

                        } else if(verb>=0.3 & verb <0.6) {
                                data.move_proposal(1)++;
                                prune_q(all_forest_q.trees[t], data, partial_residuals_q,prediction_train_sum_c);
                        } else {
                                data.move_proposal(2)++;

                                // cout << " Change error Q" << endl;
                                change_q(all_forest_q.trees[t], data, partial_residuals_q,prediction_train_sum_c);
                                // std::cout << "Error after change" << endl;
                        }

                        updateLambda(all_forest_q.trees[t],data);

                        // Getting predictions
                        // cout << " Error on Get Predictions Q" << endl;
                        getPredictions_q(all_forest_q.trees[t],data,q_hat,prediction_test_q);
                        // cout << " NEW hat{Q}: "<< q_hat(0) << endl;

                        // Updating the tree
                        // cout << "Residuals error 2.0"<< endl;
                        tree_fits_store_q.col(t) = q_hat;
                        // cout << "Residuals error 3.0"<< endl;
                        tree_fits_store_test_q.col(t) = prediction_test_q;
                        // cout << "Residuals error 4.0"<< endl;
                }



                // Summing over all trees (quality trees)
                // cout << " error here" << endl;
                prediction_train_sum_q = sum(tree_fits_store_q,1);
                prediction_test_sum_q = sum(tree_fits_store_test_q,1);


                // Need to update the Wishart matrix and alll its elements
                // cout << "Error on the wishart update" << endl;
                // cout << "Prediction value C: " << prediction_train_sum_c(0) << endl;
                // cout << "Prediction value Q: " << prediction_train_sum_q(0) << endl;

                updateP(prediction_train_sum_c,
                        prediction_train_sum_q,
                        data);

                // std::cout << " All good Q" << endl;
                if(i >= n_burn){
                        // Storing the predictions
                        c_train_hat_post.col(curr) = prediction_train_sum_c;
                        q_train_hat_post.col(curr) = prediction_train_sum_q;

                        c_test_hat_post.col(curr) = prediction_test_sum_c;
                        q_test_hat_post.col(curr) = prediction_test_sum_q;

                        tau_c_post(curr) = data.tau_c;
                        tau_q_post(curr) = data.tau_q;
                        rho_post(curr) = data.rho;
                        curr++;
                }

                // Storing all hyperparameters
                all_tau_c(i) = data.tau_c;
                all_tau_q(i) = data.tau_q;
                all_rho(i) = data.rho;

                // cout << " Here???" << endl;
                pb += 1;

        }
        // Initialising PB
        std::cout << "[";
        int k = 0;
        // Evaluating progress bar
        for(;k<=pb*width/data.n_mcmc;k++){
                std::cout << "=";
        }

        for(; k < width;k++){
                std:: cout << " ";
        }

        std::cout << "] " << std::setprecision(5) << 100 << "%\r";
        std::cout.flush();

        std::cout << std::endl;

        return Rcpp::List::create(c_train_hat_post, //[1]
                                  q_train_hat_post, //[2]
                                  c_test_hat_post, //[3]
                                  q_test_hat_post, //[4]
                                  tau_c_post, //[5]
                                  tau_q_post, //[6]
                                  rho_post, //[7]
                                  data.move_proposal, // [8]
                                  data.move_acceptance,// [9]
                                  all_tau_c, // [10]
                                  all_tau_q, //[11]
                                  all_rho); // [12]
}


//[[Rcpp::export]]
arma::mat mat_init(int n){
        arma::mat A(n,1,arma::fill::ones);
        return A + 4.0;
}


//[[Rcpp::export]]
arma::vec vec_init(int n){
        arma::vec A(n);
        return A+3.0;
}


// Comparing matrix inversions in armadillo
//[[Rcpp::export]]
arma::mat std_inv(arma::mat A, arma::vec diag){

        arma::mat diag_aux = arma::diagmat(diag);
        return arma::inv(A.t()*A+diag_aux);
}

//[[Rcpp::export]]
arma::mat std_pinv(arma::mat A, arma::vec diag){

        arma::mat diag_aux = arma::diagmat(diag);
        return arma::inv_sympd(A.t()*A+diag_aux);
}

//[[Rcpp::export]]
arma::mat faster_simple_std_inv(arma::mat A, arma::vec diag){
        arma::mat diag_aux = arma::diagmat(diag);
        arma::mat L = chol(A.t()*A+diag_aux,"lower");
        return arma::inv(L.t()*L);
}

//[[Rcpp::export]]
double log_test(double a){

        return log(a);
}


//[[Rcpp::export]]
arma::mat faster_std_inv(arma::mat A, arma::vec diag){
        arma::mat ADinvAt = A.t()*arma::diagmat(1.0/diag)*A;
        arma::mat L = arma::chol(ADinvAt + arma::eye(ADinvAt.n_cols,ADinvAt.n_cols),"lower");
        arma::mat invsqrtDA = arma::solve(A.t()/arma::diagmat(arma::sqrt(diag)),L.t());
        arma::mat Ainv = invsqrtDA *invsqrtDA.t()/(ADinvAt + arma::eye(ADinvAt.n_cols,ADinvAt.n_cols));
        return Ainv;
}


//[[Rcpp::export]]
arma::vec rMVN2(const arma::vec& b, const arma::mat& Q)
{
        arma::mat Q_inv = arma::inv(Q);
        arma::mat U = arma::chol(Q_inv, "lower");
        arma::vec z= arma::randn<arma::mat>(Q.n_cols);

        return arma::solve(U.t(), arma::solve(U, z, arma::solve_opts::no_approx), arma::solve_opts::no_approx) + b;
}





//[[Rcpp::export]]
arma::vec rMVNslow(const arma::vec& b, const arma::mat& Q){

        // cout << "Error sample BETA" << endl;
        arma::vec sample = arma::randn<arma::mat>(Q.n_cols);
        return arma::chol(Q,"lower")*sample + b;

}

//[[Rcpp::export]]
arma::mat matrix_mat(arma::cube array){
        return array.slice(1).t()*array.slice(2);
}







