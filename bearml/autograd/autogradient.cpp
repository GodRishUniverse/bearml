#include "autogradient.h"
#include "tensor/Tensor.h"

using ll = long long;
namespace bearml {
    namespace autogradient{

       // no named namespace as we do not want topological sort to be called outside
       namespace{
          template <typename T>
          std::vector<std::shared_ptr<bearml::Node<T>>> topological_sort(std::shared_ptr<bearml::Node<T>> end_node){

             std::map<std::shared_ptr<bearml::Node<T>>, ll> child_counts;
             std::vector<std::shared_ptr<bearml::Node<T>>> stack;
             stack.push_back(end_node);

             while (!stack.empty()){
                std::shared_ptr<bearml::Node<T>> node = stack[stack.size()-1]; stack.pop_back();
                auto f = child_counts.find(node);
                if (f != child_counts.end()){
                   child_counts[node] += 1;
                }else{
                   child_counts[node] = 1;
                   stack.insert(stack.end(),node->inputs.begin(),node->inputs.end()); // node->inputs are the parents
                }
             }


             std::vector<std::shared_ptr<bearml::Node<T>>> childless_nodes {end_node}; // assumption is that we do not consider the child nodes of the calling node here, even if it has some children
             std::vector<std::shared_ptr<bearml::Node<T>>> sorted;
             while (!childless_nodes.empty()){
                std::shared_ptr<bearml::Node<T>> node = childless_nodes[childless_nodes.size()-1]; childless_nodes.pop_back();
                sorted.push_back(node); // like python yield
                for (auto parent  : node->inputs){

                   if (!parent) continue; // skip null

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
       T backward(std::shared_ptr<bearml::Node<T>> end_node, bool accumulate){

          std::vector<std::shared_ptr<bearml::Node<T>>> all_nodes = topological_sort(end_node);
          // we clear grads right now
          if (!accumulate){
             for (const auto& node : all_nodes) {
                if constexpr (std::is_same<T,double>::value){
                    node->grad = 0.0;
                }else if constexpr (bearml::is_tensor_v<T>){
                    node->grad = T(node->val.getShape(), node->val.getDevice());
                }
             }
          }

          if constexpr (std::is_same<T,double>::value){
              end_node->grad = 1.0;
          }else if constexpr (bearml::is_tensor_v<T>){
              end_node->grad = T(end_node->val.getShape(), end_node->val.getDevice()); // The whole matrix is filled with 1 - because we want to compute the Jacobian matrix
              end_node->grad.fill(1.0); // fill returns void;
          }

          for (const auto& node : all_nodes) {
             // Only propagate if this node has a gradient
            if constexpr (std::is_same<T,double>::value){
                if (node->grad != 0.0 && node->backward_fn) {
                      node->backward_fn();
                }
            }else if constexpr (bearml::is_tensor_v<T>){
                if (T::has_nonzero_gradient(node->grad) && node->backward_fn) {
                      node->backward_fn();
                }
            }

          }
          return end_node->grad;
       }
    }
}

// Template specified

template double bearml::autogradient::backward<double>(std::shared_ptr<bearml::Node<double>>, bool);
template bearml::Tensor<double> bearml::autogradient::backward<bearml::Tensor<double>>(std::shared_ptr<bearml::Node<bearml::Tensor<double>>>, bool);
template bearml::Tensor<float> bearml::autogradient::backward<bearml::Tensor<float>>(std::shared_ptr<bearml::Node<bearml::Tensor<float>>>, bool);
