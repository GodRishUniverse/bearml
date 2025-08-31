#pragma once

#include <vector>
#include <numeric>
#include <stdexcept>
#include <algorithm>

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

        //works
        static std::vector<int> computeBroadcastShape(
            const std::vector<int>& A, const std::vector<int>& B)
        {
            size_t n = std::max(A.size(), B.size()); // get the maximum shape
            std::vector<int> a(A), b(B); // get two copies here
            a.insert(a.begin(), n - A.size(), 1); // expand the shape by padding with 1s
            b.insert(b.begin(), n - B.size(), 1); // expand the shape by padding withs 1s


            //TODO: USE STACK OF Operations::BroadcastOp for storing the broadcast operations which can be popped and reversed by the Reduction::Op
            // NEED TO figure out how the reduction::OP vector would be passed to the class so that reduce operations can occur as well in the autogradient Node class
            std::vector<int> out(n);// final shape outputted
            for (size_t i = 0; i < n; ++i) {
                // We compute the shape based on the idea that their shapes match or at least one of them as a 1 in that dim
                if (a[i] == b[i] || a[i] == 1 || b[i] == 1) {
                    // reverse of this is basically summation across i with keepdims=false -> but how to know which tensor -> using an if statement maybe?
                    out[i] = std::max(a[i], b[i]);
                } else {
                    throw std::invalid_argument("Shapes not broadcastable"); // both are non-1 and have different shapes
                }
            }
            return out;
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
