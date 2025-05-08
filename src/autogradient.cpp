#include <algorithm>
#include <vector>

#include "Tensor.h"
#include "Matrix.h"

using namespace std;




// Will use reverse mode -  https://en.wikipedia.org/wiki/Automatic_differentiation  
// https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation
namespace simplenet{
    // TODO: This is just a placeholder for now

    // Notes for implementation 
    /*

    We will be using a topological sort to walk the graph and compute the derivatives
    We will design a computational graph and use a topological sort

    
    https://en.wikipedia.org/wiki/Topological_sort
    
    */


    // We need to create a representation for a computation graph
    // then we can use a reverse topological order to walk the graph and compute the derivatives backward from the output
    //    x1 ---- x3 ---- out 
    //    x2-----/
    //      here we have x1, x2, x3, out - we start the backward pass from out to compute the derivatives of out with respect to all the nodes 
    // Algorithm is as follows:

    /*
    
    def gradient(out):
        node_to_grad = {out: [1]} // dictionary for node to list of gradients for each node that is connected to it
        
        for i in reverse_topological_sort(out):
            v_i_adjoint = sum(node_to_grad[i])

            for k in inputs(i):
                compute v_k_to_i_adjoint = v_i_adjoint * d_v_i / d_v_k
                append v_k_to_i_adjoint to node_to_grad[k] // propagate the partial adjoints to its inputs 
        return adjoint of input v_input(s) - this is the gradient of the function (network) with respect to all the inputs  
    */


}
namespace simplenet::autogradient{


}
