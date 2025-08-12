#include "autogradient.h"

   // TODO: implement the reverse topological order traversal
   // TODO: implement the gradient function
   // TODO: implement the backward pass (which is the gradient function) - used in the autogradient namespace
   // TODO: fix for MATRIX AND TENSOR Types
using ll = long long;
namespace simplenet::autogradient{

   // no named namespace as we do not want topological sort to be called outside
   namespace{
      template <typename T>
      std::vector<std::shared_ptr<simplenet::Node<T>>> topological_sort(std::shared_ptr<simplenet::Node<T>> end_node){
         std::map<std::shared_ptr<simplenet::Node<T>>, ll> child_counts;
         std::vector<std::shared_ptr<simplenet::Node<T>>> stack;
         stack.push_back(end_node);

         while (!stack.empty()){
            std::shared_ptr<simplenet::Node<T>> node = stack[stack.size()-1]; stack.pop_back();
            auto f = child_counts.find(node);
            if (f != child_counts.end()){
               child_counts[node] += 1;
            }else{
               child_counts[node] = 1;
               stack.insert(stack.end(),node->inputs.begin(),node->inputs.end()); // node->inputs are the parents
            }
         }


         std::vector<std::shared_ptr<simplenet::Node<T>>> childless_nodes {end_node}; // assumption is that we do not consider the child nodes of the calling node here, even if it has some children
         std::vector<std::shared_ptr<simplenet::Node<T>>> sorted;
         while (!childless_nodes.empty()){
            std::shared_ptr<simplenet::Node<T>> node = childless_nodes[childless_nodes.size()-1]; childless_nodes.pop_back();
            sorted.push_back(node); // like python yield
            for (auto parent  : node->inputs){
               if (child_counts[parent] == 1){
                  childless_nodes.push_back(parent);
               }else{
                  child_counts[parent] -= 1;
               }
            }
         }

         return sorted;
      }
   }

   // TODO: fix for MATRIX AND TENSOR TYPES - also add a boolean for gradient accumulation as well
   template <typename T>
   T backward(std::shared_ptr<simplenet::Node<T>> end_node, bool accumulate = false){

      std::vector<std::shared_ptr<simplenet::Node<T>>> all_nodes = topological_sort(end_node);
      // we clear grads right now
      if (!accumulate){
         for (auto node : all_nodes) {
            node->grad = 0.0; // TODO: change to default values - but need to figure out what shape of matrix here
         }
      }


      end_node->grad = 1.0; // TODO: change to default values - identity matrices - but need to figure out what shape of identity matrix to choose

      for (auto node : all_nodes) {
         // Only propagate if this node has a gradient
         if (node->grad != 0.0 && node->backward_fn) {
               node->backward_fn();
         }
      }

      return end_node->grad;
   }




}
