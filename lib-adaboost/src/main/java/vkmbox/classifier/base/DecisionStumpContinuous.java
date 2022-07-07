package vkmbox.classifier.base;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *  DecisionStump-Continuous classifier.
 *  Classifies: if value > threshold then sign else -1 * sign, 
 *  where sign = {-1,+1}
 */
public class DecisionStumpContinuous {
    
    private final int featureNumber;
    private final int sign;
    private final double threshold;
    
    public DecisionStumpContinuous(int featureNumber, int sign, double threshold) {
        this.featureNumber = featureNumber;
        this.sign = sign;
        this.threshold = threshold;
    }

    public int[] classify(INDArray dataX) {
        return getClassification(dataX, featureNumber, threshold, sign);
    }
    
    public INDArray classifyIND(INDArray dataX, DataType dataType) {
        int[] data = getClassification(dataX, featureNumber, threshold, sign);
        return Nd4j.create(data, new long[]{data.length}, dataType);
    }

    public static int[] getClassification
            (INDArray dataX, int featureNumber, double threshold, int sign) {
        //double[] featureX = ArrayUtil.flattenDoubleArray(dataX.getColumn(featureNumber));
        int samplesCount = dataX.rows();
        int[] result = new int[samplesCount];
        for (int idx = 0; idx < samplesCount; idx++) {
            result[idx] = dataX.getDouble(idx, featureNumber) > threshold ? sign: -1*sign;
        }
        return result; //[sign if value > threshold else -sign for value in X[:, feature_number]]
    }

    public static double getError(INDArray dataX, INDArray dataY, int featureNumber
            , /*INDArray*/double[] weights, int sign, double threshold) {
        double error = 0.0;
        int samplesCount = dataX.rows();
        for (int idx = 0; idx < samplesCount; idx++) {
            int value = dataX.getDouble(idx, featureNumber) > threshold ? sign: -1*sign;
            error += Math.abs((value-dataY.getDouble(idx))/2)*weights[idx];
        }
        return error;
    }

    @Override
    public String toString() {
        return String.format("feature_number: %d, sign: %d, threshold: %f"
                , featureNumber, sign, threshold);
    }
}
