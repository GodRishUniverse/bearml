#pragma once

#include <utility>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include "operations/op.h"


using ll = long long;

namespace simplenet {
    namespace utils {


        bool negOrZeroInSizeCheck(const std::vector<int>& sizePassedDown);

        static bool isSizeValid(const std::vector<int>& sizePassedDown, const ll& sizeOfTensor) {
            if (negOrZeroInSizeCheck(sizePassedDown)){
                return false;
            }

            ll total = 1; // cause size may be huge
            for (const int & i : sizePassedDown){
                total*= i;
            }

            return (total == sizeOfTensor);
        }

        // checks if the index is valid or not
        static bool isIndexValid(std::vector<int>& sizePassedDown, const std::vector<int>& shape) {
            for (size_t i = 0 ; i < shape.size(); i++){
                if (sizePassedDown[i] <0 || sizePassedDown[i] >= shape[i]){
                    return false;
                }
            }
            return true;
        }

        //TODO: rethink on whether the BroadCastOp is correctly inverted to ReductionOp  or not in computeBroadcastShape
        static std::pair< std::vector<int>, std::pair< std::vector<int>, std::vector<int>>> computeBroadcastShape(
            const std::vector<int>& A, const std::vector<int>& B)
        {
            size_t n = std::max(A.size(), B.size()); // get the maximum shape
            std::vector<int> a(A), b(B); // get two copies here

            std::vector<Operations::BroadcastOp> a_stack;
            std::vector<Operations::BroadcastOp> b_stack;


            a.insert(a.begin(), n - A.size(), 1); // expand the shape by padding with 1s
            for (size_t i =0; i < n-A.size(); i++){
                Operations::BroadcastOp op = Operations::BroadcastOp(Operations::BroadcastOp::PAD, i, 0,1);
                a_stack.push_back(op);
            }
            b.insert(b.begin(), n - B.size(), 1); // expand the shape by padding withs 1s
            for (size_t i =0; i < n-B.size(); i++){
                Operations::BroadcastOp op = Operations::BroadcastOp(Operations::BroadcastOp::PAD, i, 0,1);
                b_stack.push_back(op);
            }

            // TODO: USE STACK OF Operations::BroadcastOp for storing the broadcast operations which can be popped and reversed by the Reduction::Op
            // NEED TO figure out how the reduction::OP vector would be passed to the class so that reduce operations can occur as well in the autogradient Node class
            std::vector<int> out(n);// final shape outputted
            for (size_t i = 0; i < n; ++i) {
                // We compute the shape based on the idea that their shapes match or at least one of them as a 1 in that dim
                if (a[i] == b[i] || a[i] == 1 || b[i] == 1) {
                    // reverse of this is basically summation across i with keepdims=false -> but how to know which tensor -> using an if statement maybe?
                    out[i] = std::max(a[i], b[i]);
                    if (a[i] != out[i]){
                        Operations::BroadcastOp op_a = Operations::BroadcastOp(Operations::BroadcastOp::BROADCAST, i, a[i],out[i]);
                        a_stack.push_back(op_a);
                    }
                    if (b[i] != out[i]){
                        Operations::BroadcastOp op_b = Operations::BroadcastOp(Operations::BroadcastOp::BROADCAST, i, b[i],out[i]);
                        b_stack.push_back(op_b);
                    }
                } else {
                    throw std::invalid_argument("Shapes not broadcastable"); // both are non-1 and have different shapes
                }
            }
            // Construct reduction OPs order
            std::vector<Operations::ReductionOp> ops_to_execute_a;
            std::vector<Operations::ReductionOp> ops_to_execute_b;

            while(!a_stack.empty()){
                Operations::BroadcastOp op = a_stack[a_stack.size()-1]; a_stack.pop_back();
                ops_to_execute_a.push_back(Operations::ReductionOp::convert(op));
            }

            while(!b_stack.empty()){
                Operations::BroadcastOp op = b_stack[b_stack.size()-1]; b_stack.pop_back();
                ops_to_execute_b.push_back(Operations::ReductionOp::convert(op));
            }


            return {out, {a,b}}; // need to return both lists of ops to execute
        }

        static std::vector<int> computeBroadcastStrides(
            const std::vector<int>&   origShape,
            const std::vector<int>&    origStrides,
            const std::vector<int>&   targetShape)
        {
            size_t n = targetShape.size();
            // pad on the left
            std::vector<int>  s = origShape;
            std::vector<int>  st = origStrides;
            s.insert(s.begin(), n - s.size(), 1);
            st.insert(st.begin(), n - st.size(), 0);

            std::vector<int> out(n);
            for (size_t i = 0; i < n; ++i) {
                out[i] = (s[i] == targetShape[i] ? st[i] : 0);
            }
            return out;
        }


    }
}
