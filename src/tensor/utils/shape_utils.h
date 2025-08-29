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
            size_t n = std::max(A.size(), B.size());
            std::vector<int> a(A), b(B);
            a.insert(a.begin(), n - A.size(), 1);
            b.insert(b.begin(), n - B.size(), 1);

            std::vector<int> out(n);
            for (size_t i = 0; i < n; ++i) {
                if (a[i] == b[i] || a[i] == 1 || b[i] == 1) {
                    out[i] = std::max(a[i], b[i]);
                } else {
                    throw std::invalid_argument("Shapes not broadcastable");
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
