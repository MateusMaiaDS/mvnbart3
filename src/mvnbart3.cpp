#include "mvnbart3.h"
#include <iomanip>
#include<cmath>
#include <random>
#include <RcppArmadillo.h>
using namespace std;

// =====================================
// Statistics Function
// =====================================

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
                        arma::mat y_mat_,
                        arma::mat x_test_,
                        arma::mat x_cut_,
                        int n_tree_,
                        int node_min_size_,
                        double alpha_,
                        double beta_,
                        double nu_,
                        arma::vec sigma_mu_,
                        arma::mat Sigma_,
                        arma::mat S_0_wish_,
                        arma::vec a_j_vec_,
                        arma::vec A_j_vec_,
                        double n_mcmc_,
                        double n_burn_){


        // Assign the variables
        x_train = x_train_;
        y_mat = y_mat_;
        x_test = x_test_;
        xcut = x_cut_;
        n_tree = n_tree_;
        node_min_size = node_min_size_;
        alpha = alpha_;
        beta = beta_;
        nu = nu_;
        sigma_mu = sigma_mu_;

        Sigma = Sigma_;
        S_0_wish = S_0_wish_;
        a_j_vec = a_j_vec_;
        A_j_vec = A_j_vec_;
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
        var_split_rule = -1.0;
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
     left -> var_split_rule = -1.0;
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
     right -> var_split_rule = -1.0;
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
        trees.resize(data.n_tree*data.y_mat.n_cols);
        for(int  i=0;i<(data.n_tree*data.y_mat.n_cols);i++){
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
void grow(Node* tree, modelParam &data, arma::vec &curr_res, arma::vec& curr_u){

        // Getting the number of terminal nodes
        std::vector<Node*> t_nodes = leaves(tree) ;
        std::vector<Node*> nog_nodes = nogs(tree);

        // Selecting one node to be sampled
        Node* g_node = sample_node(t_nodes);

        // Store all old quantities that will be used or not
        int old_var_split = g_node->var_split;
        int old_var_split_rule = g_node->var_split_rule;

        // // Calculating the whole likelihood fo the tree
        for(int i = 0; i < t_nodes.size(); i++){
                // cout << "Error gpNodeLogLike" << endl;
                t_nodes[i]->updateResiduals(data, curr_res,curr_u); // Do I need to do this?
        }

        // Calculating the likelihood of the current grown node
        g_node->nodeLogLike(data);
        // cout << "LogLike Node ok Grow" << endl;

        // Adding the leaves
        g_node->addingLeaves(data);

        bool no_valid_node = false;
        int p_try = 0;

        // Trying to find a cutpoint
        arma::vec split_candidates = arma::shuffle(arma::regspace(0,1,data.x_train.n_cols-1));
        Rcpp::NumericVector valid_cutpoint;

        while(!no_valid_node){
                g_node->var_split = split_candidates(p_try);

                Rcpp::NumericVector var_split_range;

                // Getting the maximum and the minimum;
                for(int i = 0; i < g_node->n_leaf; i++){
                        var_split_range.push_back(data.x_train(g_node->train_index[i],g_node->var_split));
                }

                // Getting the minimum and the maximum;
                double max_rule = max(var_split_range);
                double min_rule = min(var_split_range);

                for(int cut = 0; cut < data.xcut.n_rows; cut++ ){
                        if((data.xcut(cut,g_node->var_split)>min_rule) & (data.xcut(cut,g_node->var_split)<max_rule)){
                                valid_cutpoint.push_back(data.xcut(cut,g_node->var_split));
                        }
                }

                if(valid_cutpoint.size()==0){
                        p_try++;
                        if(p_try>=data.x_train.n_cols){
                                no_valid_node = true;
                        };
                } else {
                        break; // Go out from the while
                }
        }

        if(no_valid_node){
        // Returning to the old values
                g_node->var_split = old_var_split;
                g_node->var_split_rule = old_var_split_rule;

                g_node->deletingLeaves();
                return;
        }

        // Selecting a rule (here I'm actually selecting the var split rule);
        g_node->var_split_rule = valid_cutpoint[arma::randi(arma::distr_param(0,valid_cutpoint.size()))];
        // cout << "The current var split rule is: " << g_node->var_split_rule << endl;

        // Create an aux for the left and right index
        int train_left_counter = 0;
        int train_right_counter = 0;

        int test_left_counter = 0;
        int test_right_counter = 0;

        // Create a matrix to store the new x_{j} and then get information of min and max
        arma::mat x_new_left(g_node->n_leaf, data.x_train.n_cols);
        arma::mat x_new_right(g_node->n_leaf,data.x_train.n_cols);

        // Updating the left and the right nodes
        for(int i = 0;i<data.x_train.n_rows;i++){
                if(g_node -> train_index[i] == -1 ){
                        g_node->left->n_leaf = train_left_counter;
                        g_node->right->n_leaf = train_right_counter;
                        break;
                }
                if(data.x_train(g_node->train_index[i],g_node->var_split)<=g_node->var_split_rule){
                        g_node->left->train_index[train_left_counter] = g_node->train_index[i];
                        x_new_left.row(train_left_counter) = data.x_train.row(g_node->train_index[i]);
                        train_left_counter++;
                } else {
                        g_node->right->train_index[train_right_counter] = g_node->train_index[i];
                        x_new_right.row(train_right_counter) = data.x_train.row(g_node->train_index[i]);
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
                if(data.x_test(g_node->test_index[i],g_node->var_split)<=g_node->var_split_rule){
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


                g_node->deletingLeaves();
                return;
        }




        // Updating the loglikelihood for those terminal nodes
        // cout << "Calculating likelihood of the new node on left" << endl;
        // cout << " ACCEPTED" << endl;
        g_node->left->updateResiduals(data,curr_res,curr_u);
        g_node->right->updateResiduals(data,curr_res,curr_u);


        // Calculating the likelihood on the new node on the left
        g_node->left->nodeLogLike(data);
        // cout << "Calculating likelihood of the new node on right" << endl;
        g_node->right->nodeLogLike(data);
        // cout << "NodeLogLike ok again" << endl;


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
        double acceptance = exp(new_tree_log_like  + log_transition_prob + tree_prior);


        // Keeping the new tree or not
        if(arma::randu(arma::distr_param(0.0,1.0)) < acceptance){
                // Do nothing just keep the new tree
                data.move_acceptance(0)++;
        } else {
                // Returning to the old values
                g_node->var_split = old_var_split;
                g_node->var_split_rule = old_var_split_rule;

                g_node->deletingLeaves();
        }

        return;

}


// Pruning a tree
void prune(Node* tree, modelParam&data, arma::vec &curr_res, arma::vec &curr_u){


        // Getting the number of terminal nodes
        std::vector<Node*> t_nodes = leaves(tree);

        // Can't prune a root
        if(t_nodes.size()==1){
                // cout << "Nodes size " << t_nodes.size() <<endl;
                t_nodes[0]->updateResiduals(data,curr_res,curr_u);
                t_nodes[0]->nodeLogLike(data);
                return;
        }

        std::vector<Node*> nog_nodes = nogs(tree);

        // Selecting one node to be sampled
        Node* p_node = sample_node(nog_nodes);


        // Calculating the whole likelihood fo the tree
        for(int i = 0; i < t_nodes.size(); i++){
                t_nodes[i]->updateResiduals(data, curr_res,curr_u);
        }

        // cout << "Error C1" << endl;
        // Updating the loglikelihood of the selected pruned node
        p_node->updateResiduals(data,curr_res,curr_u);
        p_node->nodeLogLike(data);
        p_node->left->nodeLogLike(data);
        p_node->right->nodeLogLike(data);

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
                data.move_acceptance(2)++;
        } else {
                // p_node->left->gpNodeLogLike(data, curr_res);
                // p_node->right->gpNodeLogLike(data, curr_res);
        }

        return;
}


// // Creating the change verb
void change(Node* tree, modelParam &data, arma::vec &curr_res, arma::vec &curr_u){


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
                t_nodes[i]->updateResiduals(data, curr_res,curr_u);
        }

        // Calculating the loglikelihood of the old nodes
        c_node->left->nodeLogLike(data);
        c_node->right->nodeLogLike(data);


        // Storing all the old loglikelihood from left
        double old_left_log_like = c_node->left->log_likelihood;
        double old_left_r_sum = c_node->left->r_sum;
        double old_left_u_sum = c_node->left->u_sum;
        double old_left_Gamma_j = c_node->left->Gamma_j;
        double old_left_S_j = c_node->left->S_j;

        arma::vec old_left_train_index = c_node->left->train_index;
        c_node->left->train_index.fill(-1); // Returning to the original
        int old_left_n_leaf = c_node->left->n_leaf;


        // Storing all of the old loglikelihood from right;
        double old_right_log_like = c_node->right->log_likelihood;
        double old_right_r_sum = c_node->right->r_sum;
        double old_right_u_sum = c_node->right->u_sum;
        double old_right_Gamma_j = c_node->right->Gamma_j;
        double old_right_S_j = c_node->right->S_j;


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

        // Choosing only valid cutpoints;
        bool no_valid_node = false;
        int p_try = 0;

        // Trying to find a cutpoint
        arma::vec split_candidates = arma::shuffle(arma::regspace(0,1,data.x_train.n_cols-1));
        Rcpp::NumericVector valid_cutpoint;

        while(!no_valid_node){
                c_node->var_split = split_candidates(p_try);

                Rcpp::NumericVector var_split_range;

                // Getting the maximum and the minimum;
                for(int i = 0; i < c_node->n_leaf; i++){
                        var_split_range.push_back(data.x_train(c_node->train_index[i],c_node->var_split));
                }

                // Getting the minimum and the maximum;
                double max_rule = max(var_split_range);
                double min_rule = min(var_split_range);

                for(int cut = 0; cut < data.xcut.n_rows; cut++ ){
                        if((data.xcut(cut,c_node->var_split)>min_rule) & (data.xcut(cut,c_node->var_split)<max_rule)){
                                valid_cutpoint.push_back(data.xcut(cut,c_node->var_split));
                        }
                }

                if(valid_cutpoint.size()==0){
                        p_try++;
                        if(p_try>=data.x_train.n_cols){
                                no_valid_node = true;
                        };
                } else {
                        break; // Go out from the while
                }
        }

        if(no_valid_node){
                // Returning to the old values
                c_node->var_split = old_var_split;
                c_node->var_split_rule = old_var_split_rule;
                return;
        }

        // Selecting a rule (here I'm actually selecting the var split rule);
        c_node->var_split_rule = valid_cutpoint[arma::randi(arma::distr_param(0,valid_cutpoint.size()))];

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

                if(data.x_train(c_node->train_index[i],c_node->var_split)<=c_node->var_split_rule){
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

                if(data.x_test(c_node->test_index[i],c_node->var_split)<=c_node->var_split_rule){
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
                c_node->left->r_sum = old_left_r_sum;
                c_node->left->u_sum = old_left_u_sum;
                c_node->left->Gamma_j = old_left_Gamma_j ;
                c_node->left->S_j = old_left_S_j;

                c_node->left->n_leaf = old_left_n_leaf;
                c_node->left->n_leaf_test = old_left_n_leaf_test;
                c_node->left->log_likelihood = old_left_log_like;
                c_node->left->train_index = old_left_train_index;
                c_node->left->test_index = old_left_test_index;

                // Returning to the old ones
                c_node->right->r_sum = old_right_r_sum;
                c_node->right->u_sum = old_right_u_sum;
                c_node->right->Gamma_j = old_right_Gamma_j ;
                c_node->right->S_j = old_right_S_j;

                c_node->right->n_leaf = old_right_n_leaf;
                c_node->right->n_leaf_test = old_right_n_leaf_test;
                c_node->right->log_likelihood = old_right_log_like;
                c_node->right->train_index = old_right_train_index;
                c_node->right->test_index = old_right_test_index;

                return;
        }

        // Updating the new left and right loglikelihoods
        c_node->left->updateResiduals(data,curr_res,curr_u);
        c_node->right->updateResiduals(data,curr_res,curr_u);
        c_node->left->nodeLogLike(data);
        c_node->right->nodeLogLike(data);

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
                c_node->left->r_sum = old_left_r_sum;
                c_node->left->u_sum = old_left_u_sum;
                c_node->left->Gamma_j = old_left_Gamma_j ;
                c_node->left->S_j = old_left_S_j;

                c_node->left->n_leaf = old_left_n_leaf;
                c_node->left->n_leaf_test = old_left_n_leaf_test;
                c_node->left->log_likelihood = old_left_log_like;
                c_node->left->train_index = old_left_train_index;
                c_node->left->test_index = old_left_test_index;

                // Returning to the old ones
                c_node->right->r_sum = old_right_r_sum;
                c_node->right->u_sum = old_right_u_sum;
                c_node->right->Gamma_j = old_right_Gamma_j ;
                c_node->right->S_j = old_right_S_j;

                c_node->right->n_leaf = old_right_n_leaf;
                c_node->right->n_leaf_test = old_right_n_leaf_test;
                c_node->right->log_likelihood = old_right_log_like;
                c_node->right->train_index = old_right_train_index;
                c_node->right->test_index = old_right_test_index;

        }

        return;
}




// Calculating the Loglilelihood of a node
void Node::updateResiduals(modelParam& data,
                           arma::vec &curr_res,
                           arma::vec &curr_u){

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
                r_sum = 0;
                log_likelihood = -2000000; // Absurd value avoid this case
                // cout << "OOOPS something happened" << endl;
                return;
        }

        // If is smaller then the node size still need to update the quantities;
        // cout << "Node min size: " << data.node_min_size << endl;
        if(n_leaf < data.node_min_size){
                // log_likelihood = -2000000; // Absurd value avoid this case
                // cout << "OOOPS something happened" << endl;
                return;
        }

        r_sum = 0.0;
        u_sum = 0.0;

        // Train elements
        for(int i = 0; i < n_leaf;i++){
                r_sum = r_sum + curr_res(train_index[i]);
                u_sum = u_sum + curr_u(train_index[i]);
        }

        sigma_mu_j_sq = data.sigma_mu_j*data.sigma_mu_j;
        Gamma_j  = n_leaf+data.v_j/sigma_mu_j_sq;
        S_j = r_sum-u_sum;

        return;

}

void Node::nodeLogLike(modelParam& data){
        // Getting the log-likelihood;
        // double sigma_mu_j_sq = data.sigma_mu_j*data.sigma_mu_j;

        log_likelihood = -0.5*log(2*arma::datum::pi*sigma_mu_j_sq)+0.5*log(data.v_j/Gamma_j) +0.5*(S_j*S_j)/(data.v_j*Gamma_j);
        return;
}

// UPDATING MU
void updateMu(Node* tree, modelParam &data){

        // Getting the terminal nodes
        std::vector<Node*> t_nodes = leaves(tree);


        // Iterating over the terminal nodes and updating the beta values
        for(int i = 0; i < t_nodes.size();i++){
                t_nodes[i]->mu = R::rnorm((t_nodes[i]->S_j)/(t_nodes[i]->Gamma_j),sqrt(data.v_j/(t_nodes[i]->Gamma_j))) ;

        }
}

void updateSigma(arma::mat &y_mat_hat,
                 modelParam &data){

        arma::mat S(data.y_mat.n_cols,data.y_mat.n_cols,arma::fill::zeros);
        arma::mat residuals_mat = y_mat_hat-data.y_mat;
        S = residuals_mat.t()*residuals_mat;

        // Updating sigma
        data.Sigma = arma::iwishrnd((data.S_0_wish+S),data.nu+data.y_mat.n_rows);

}




// Get the prediction
// (MOST IMPORTANT AND COSTFUL FUNCTION FROM GP-BART)
void getPredictions(Node* tree,
                    modelParam &data,
                    arma::vec& current_prediction_train,
                    arma::vec& current_prediction_test){

        // Getting the current prediction
        vector<Node*> t_nodes = leaves(tree);
        for(int i = 0; i<t_nodes.size();i++){

                // Skipping empty nodes
                if(t_nodes[i]->n_leaf==0){
                        Rcpp::Rcout << " THERE ARE EMPTY NODES" << endl;
                        continue;
                }


                // For the training samples
                for(int j = 0; j<data.x_train.n_rows; j++){

                        if((t_nodes[i]->train_index[j])==-1){
                                break;
                        }
                        current_prediction_train[t_nodes[i]->train_index[j]] = t_nodes[i]->mu;
                }

                if(t_nodes[i]->n_leaf_test == 0 ){
                        continue;
                }



                // Regarding the test samples
                for(int j = 0; j< data.x_test.n_rows;j++){

                        if(t_nodes[i]->test_index[j]==-1){
                                break;
                        }

                        current_prediction_test[t_nodes[i]->test_index[j]] = t_nodes[i]->mu;
                }

        }
}


void update_a_j(modelParam &data){

        double shape_j = 0.5*(data.y_mat.n_rows+data.nu);
        arma::mat Precision = arma::inv(data.Sigma);

        // Rcpp::Rcout << "a_j_vec is: "<< data.a_j_vec.size() << endl;
        // Rcpp::Rcout << "A_j_vec is: "<< data.a_j_vec.size() << endl;
        // Rcpp::Rcout << "S_0_vec is: "<< data.S_0_wish.size() << endl;

        // Calcularting shape and scale parameters
        for(int j = 0; j < data.y_mat.n_cols; j++){
                double scale_j = 1/(data.A_j_vec(j)*data.A_j_vec(j))+data.nu*Precision(j,j);
                double a_j_vec_double_aux = R::rgamma(shape_j,1/scale_j);
                data.a_j_vec(j) = 1/a_j_vec_double_aux;
                data.S_0_wish(j,j) = (2*data.nu)/data.a_j_vec(j);
                // Rcpp::Rcout << " Iteration j" << endl;
        }

        return;
}

// Creating the BART function
// [[Rcpp::export]]
Rcpp::List cppbart(arma::mat x_train,
          arma::mat y_mat,
          arma::mat x_test,
          arma::mat x_cut,
          int n_tree,
          int node_min_size,
          int n_mcmc,
          int n_burn,
          arma::mat Sigma_init,
          arma::vec mu_init,
          arma::vec sigma_mu,
          double alpha, double beta, double nu,
          arma::mat S_0_wish,
          arma::vec A_j_vec,
          arma::vec a_j_vec){

        // Posterior counter
        int curr = 0;



        // cout << " Error on model.param" << endl;
        // Creating the structu object
        modelParam data(x_train,
                        y_mat,
                        x_test,
                        x_cut,
                        n_tree,
                        node_min_size,
                        alpha,
                        beta,
                        nu,
                        sigma_mu,
                        Sigma_init,
                        S_0_wish,
                        A_j_vec,
                        a_j_vec,
                        n_mcmc,
                        n_burn);

        // Getting the n_post
        int n_post = n_mcmc - n_burn;

        // Rcpp::Rcout << "error here" << endl;

        // Defining those elements
        arma::cube y_train_hat_post(data.y_mat.n_rows,data.y_mat.n_cols,n_post,arma::fill::zeros);
        arma::cube y_test_hat_post(data.x_test.n_rows,data.y_mat.n_cols,n_post,arma::fill::zeros);
        arma::cube Sigma_post(data.Sigma.n_rows,data.Sigma.n_cols,n_post,arma::fill::zeros);
        arma::cube all_Sigma_post(data.Sigma.n_rows,data.Sigma.n_cols,n_mcmc,arma::fill::zeros);

        // Rcpp::Rcout << "error here2" << endl;

        // =====================================
        // For the moment I will not store those
        // =====================================
        // arma::cube all_tree_post(y_mat.size(),n_tree,n_post,arma::fill::zeros);


        // Defining other variables
        // arma::vec partial_pred = arma::mat(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
        arma::vec partial_residuals(data.x_train.n_rows,arma::fill::zeros);
        arma::cube tree_fits_store(data.x_train.n_rows,data.n_tree,data.y_mat.n_cols,arma::fill::zeros);
        arma::cube tree_fits_store_test(data.x_test.n_rows,data.n_tree,y_mat.n_cols,arma::fill::zeros);

        // Rcpp::Rcout << "error here3" << endl;

        // In case if I need to start with another initial values
        // for(int i = 0 ; i < data.n_tree ; i ++ ){
        //         tree_fits_store.col(i) = partial_pred;
        // }

        double verb;

        // Defining progress bars parameters
        const int width = 70;
        double pb = 0;


        // cout << " Error one " << endl;

        // Selecting the train
        Forest all_forest(data);

        // Creating variables to help define in which tree set we are;
        int curr_tree_counter;

        for(int i = 0;i<data.n_mcmc;i++){

                // Initialising PB
                Rcpp::Rcout << "[";
                int k = 0;
                // Evaluating progress bar
                for(;k<=pb*width/data.n_mcmc;k++){
                        Rcpp::Rcout << "=";
                }

                for(; k < width;k++){
                        Rcpp::Rcout << " ";
                }

                Rcpp::Rcout << "] " << std::setprecision(5) << (pb/data.n_mcmc)*100 << "%\r";
                Rcpp::Rcout.flush();


                // Getting zeros
                arma::mat prediction_train_sum(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
                arma::mat prediction_test_sum(data.x_test.n_rows,data.y_mat.n_cols,arma::fill::zeros);

                // Matrix that store all the predictions for all y
                arma::mat y_mat_hat(data.x_train.n_rows,data.y_mat.n_cols,arma::fill::zeros);
                arma::mat y_mat_test_hat(data.x_test.n_rows,data.y_mat.n_cols,arma::fill::zeros);

                // Iterating over the d-dimension MATRICES of the response.
                for(int j = 0; j < data.y_mat.n_cols; j++){


                        arma::mat Sigma_j_mj(1,(data.y_mat.n_cols-1),arma::fill::zeros);
                        arma::mat Sigma_mj_j((data.y_mat.n_cols-1),1,arma::fill::zeros);
                        arma::mat Sigma_mj_mj = data.Sigma;


                        double Sigma_j_j = data.Sigma(j,j);

                        int aux_j_counter = 0;

                        // Rcpp::Rcout << "error here 3.5" << endl;

                        // Dropping the column with respect to "j"
                        Sigma_mj_mj.shed_row(j);
                        Sigma_mj_mj.shed_col(j);


                        arma::vec partial_u(data.x_train.n_rows);
                        arma::mat y_mj(data.x_train.n_rows,data.y_mat.n_cols);
                        arma::mat y_hat_mj(data.x_train.n_rows,data.y_mat.n_cols);

                        // cout << "Sigma - nrows: "<< Sigma_j_mj.n_rows;
                        // cout << "Sigma - ncols: "<< Sigma_j_mj.n_cols;

                        for(int d = 0; d < data.y_mat.n_cols; d++){

                                // Rcpp::Rcout <<  "AUX_J: " << data.Sigma(j,d) << endl;
                                if(d!=j){
                                        Sigma_j_mj(0,aux_j_counter)  = data.Sigma(j,d);
                                        Sigma_mj_j(aux_j_counter,0) = data.Sigma(j,d);
                                        aux_j_counter = aux_j_counter + 1;
                                }

                                // Rcpp::Rcout << " aux_j: " << (data.y_mat.n_cols-1) << endl;



                        }


                        // Rcpp::Rcout << "error here 3.8" << endl;

                        // ============================================
                        // This step does not iterate over the trees!!!
                        // ===========================================
                        y_mj = data.y_mat;
                        y_mj.shed_col(j);
                        y_hat_mj = y_mat_hat;
                        y_hat_mj.shed_col(j);

                        // Calculating the invertion that gonna be used for the U and V
                        arma::mat Sigma_mj_mj_inv = arma::inv(Sigma_mj_mj);

                        // Calculating the current partial U
                        for(int i_train = 0; i_train < data.y_mat.n_rows;i_train++){
                                partial_u(i_train) = arma::as_scalar(Sigma_mj_j*Sigma_mj_mj_inv*(y_mj.row(i_train)-y_hat_mj.row(i_train)));
                        }

                        double v = Sigma_j_j - arma::as_scalar(Sigma_j_mj*Sigma_mj_mj_inv*Sigma_mj_j);
                        data.v_j = v;
                        data.sigma_mu_j = data.sigma_mu(j);

                        // Rcpp::Rcout << "error here 4" << endl;


                        // Updating the tree
                                for(int t = 0; t<data.n_tree;t++){

                                        // Current tree counter
                                        curr_tree_counter = t + j*data.n_tree;
                                        // cout << "curr_tree_counter value:" << curr_tree_counter << endl;
                                        // Creating the auxliar prediction vector
                                        arma::vec y_j_hat(data.y_mat.n_rows,arma::fill::zeros);
                                        arma::vec y_j_test_hat(data.x_test.n_rows,arma::fill::zeros);

                                        // Updating the partial residuals
                                        if(data.n_tree>1){
                                                partial_residuals = data.y_mat.col(j)-sum_exclude_col(tree_fits_store.slice(j),t);
                                        } else {
                                                partial_residuals = data.y_mat.col(j);
                                        }

                                        // Iterating over all trees
                                        verb = arma::randu(arma::distr_param(0.0,1.0));

                                        if(all_forest.trees[curr_tree_counter]->isLeaf & all_forest.trees[curr_tree_counter]->isRoot){
                                                // verb = arma::randu(arma::distr_param(0.0,0.3));
                                                verb = 0.1;
                                        }

                                        // Selecting the verb
                                        if(verb < 0.25){
                                                data.move_proposal(0)++;
                                                // cout << " Grow error" << endl;
                                                // Rcpp::stop("STOP ENTERED INTO A GROW");
                                                grow(all_forest.trees[curr_tree_counter],data,partial_residuals,partial_u);
                                        } else if(verb>=0.25 & verb <0.5) {
                                                data.move_proposal(1)++;
                                                // Rcpp::stop("STOP ENTERED INTO A PRUNE");
                                                // cout << " Prune error" << endl;
                                                prune(all_forest.trees[curr_tree_counter], data, partial_residuals,partial_u);
                                        } else {
                                                data.move_proposal(2)++;
                                                // cout << " Change error" << endl;
                                                change(all_forest.trees[curr_tree_counter], data, partial_residuals,partial_u);
                                                // std::cout << "Error after change" << endl;
                                        }


                                        updateMu(all_forest.trees[curr_tree_counter],data);

                                        // Getting predictions
                                        // cout << " Error on Get Predictions" << endl;
                                        getPredictions(all_forest.trees[curr_tree_counter],data,y_j_hat,y_j_test_hat);

                                        // Updating the tree
                                        // cout << "Residuals error 2.0"<< endl;
                                        tree_fits_store.slice(j).col(t) = y_j_hat;
                                        // cout << "Residuals error 3.0"<< endl;
                                        tree_fits_store_test.slice(j).col(t) = y_j_test_hat;
                                        // cout << "Residuals error 4.0"<< endl;


                                } // End of iterations over "t"

                                // Summing over all trees
                                prediction_train_sum = sum(tree_fits_store.slice(j),1);
                                y_mat_hat.col(j) = prediction_train_sum;

                                prediction_test_sum = sum(tree_fits_store_test.slice(j),1);
                                y_mat_test_hat.col(j) = prediction_test_sum;

                }// End of iterations over "j"



                // std::cout << "Error Tau: " << data.tau<< endl;
                update_a_j(data);
                updateSigma(y_mat_hat, data);
                all_Sigma_post.slice(i) = data.Sigma;

                // std::cout << " All good " << endl;
                if(i >= n_burn){
                        // Storing the predictions
                        y_train_hat_post.slice(curr) = y_mat_hat;
                        y_test_hat_post.slice(curr) = y_mat_test_hat;
                        Sigma_post.slice(curr) = data.Sigma;
                        curr++;
                }

                pb += 1;

        }
        // Initialising PB
        Rcpp::Rcout << "[";
        int k = 0;
        // Evaluating progress bar
        for(;k<=pb*width/data.n_mcmc;k++){
                Rcpp::Rcout << "=";
        }

        for(; k < width;k++){
                Rcpp::Rcout << " ";
        }

        Rcpp::Rcout << "] " << std::setprecision(5) << 100 << "%\r";
        Rcpp::Rcout.flush();

        Rcpp::Rcout << std::endl;

        return Rcpp::List::create(y_train_hat_post, //[1]
                                  y_test_hat_post, //[2]
                                  Sigma_post, //[3]
                                  all_Sigma_post, // [4]
                                  data.move_proposal, // [5]
                                  data.move_acceptance //[6]
                                );
}


