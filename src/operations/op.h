#pragma once

#ifndef OPERATIONS
#define OPERATIONS

namespace simplenet {
    namespace Operations {

        // REDUCTION AND BROADCAST OPS
        // BroadcastOp::PAD is reversed by ReductionOp::FLATTEN
        // BroadcastOp::BROADCAST is reversed by ReductionOp::SUM

        struct BroadcastOp {
            enum Type { BROADCAST, PAD } type;
            int dimIndex, originalSize, targetSize;
        };


        struct ReductionOp {
            enum Type { FLATTEN, SUM, MEAN } type;
            int dimIndex;
            bool keepdim = true; // we would not be doing this

            // TODO: this might change as we dont pass in originalSize and targetSize
            static ReductionOp convert(BroadcastOp broadcastOp){
                ReductionOp op;
                if (broadcastOp.type == BroadcastOp::PAD){
                    op.type = FLATTEN;
                }else{
                    op.type = SUM;
                }
                op.dimIndex = broadcastOp.dimIndex;
                return op;
            }
        };
    }
}
#endif
