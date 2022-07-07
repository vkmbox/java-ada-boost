package vkmbox.classifier.util;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

public class RademacherUtil {

    public static double calculateForBiclassifiers(INDArray dataX, long subsetSize) {
        int samplesCount = dataX.rows(), featuresCount = dataX.columns();
        INDArray sorted = Nd4j.sortWithIndices(dataX.dup(), 0, true)[0];
        //ind = np.argsort(X, axis=0)
        INDArray vectorsH = Nd4j.zeros(DataType.INT32, 2*samplesCount*featuresCount, samplesCount);
        for (int feature = 0; feature < featuresCount; feature++) {
            int[] sortedColumn = sorted.getColumn(feature).toIntVector();
            for (int sample = 0; sample < samplesCount; sample++) {
                //indices = indexes.g [0:sample,feature]
                //vector_H = np.array([-1]*samples_count)
                //vector_H[indices] = 1
                INDArray vectorH = Nd4j.ones(DataType.INT32, samplesCount);
                for (int pos = 0; pos < sample; pos++) {
                    vectorH.putScalar(sortedColumn[pos], -1);
                }
                vectorsH.putRow(2*samplesCount*feature+2*sample, vectorH);
                vectorsH.putRow(2*samplesCount*feature+2*sample+1, vectorH.mul(-1));
                //vectors_H.append(vector_H)
                //vectors_H.append(-1*vector_H)
            }
        }
        double empericalSum = 0.0, subsetsNumber = Math.pow(2.0, samplesCount);
        long[] dimensionsRad = new long[]{samplesCount};
        if (subsetsNumber > subsetSize) {
            for (int dummy = 0; dummy < subsetSize; dummy++) {//in range(subset_size):
                //INDArray vectorRad = [1 if random.random() < 0.5 else -1 for _ in range(samples_count)]
                int[] buffer = new int[samplesCount];
                for (int pos = 0; pos < samplesCount; pos++) {
                    buffer[pos] = Math.random() < 0.5? -1: 1;
                }
                empericalSum += calculateMaxOnRademacherVector(vectorsH, Nd4j.create(buffer, dimensionsRad, DataType.INT32));
            }
        } else {
            for (int order = 0; order < subsetsNumber; order++) {
                //vector_rad = -1 * np.ones(samples_count, np.int32)
                StringBuilder binaryBuffer = new StringBuilder(Integer.toBinaryString(order));
                while (binaryBuffer.length() < samplesCount) {
                    binaryBuffer.insert(0, "0");
                }
                int[] buffer = new int[samplesCount];
                for (int pos = 0; pos < samplesCount; pos++) {
                    buffer[pos] = binaryBuffer.charAt(pos) == '1'? 1 : -1;
                }
                empericalSum += calculateMaxOnRademacherVector(vectorsH, Nd4j.create(buffer, dimensionsRad, DataType.INT32));
            }
        }
        return empericalSum/Math.min(subsetSize, subsetsNumber);
    }
    
    public static double calculateMaxOnRademacherVector(INDArray vectorsH, INDArray vectorRad) {
        return vectorsH.mmul(vectorRad).maxNumber().doubleValue()/vectorsH.columns();
    }
}
