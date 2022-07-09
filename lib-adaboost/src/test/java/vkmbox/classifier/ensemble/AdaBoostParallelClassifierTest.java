package vkmbox.classifier.ensemble;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import vkmbox.classifier.adaboost.AdaBoostParallelClassifier;

@Slf4j
class AdaBoostParallelClassifierTest {
    
    @Test 
    void nd4Call() {
        INDArray dataX = Nd4j.create(new double[][]
                {{0.6476239, -0.81753611, -1.61389785, -0.21274028}
                ,{-2.37482060,  0.82768797, -0.38732682, -0.30230275}
                ,{1.51783379,  1.22140561, -0.51080514, -1.18063218}
                ,{-0.98740462,  0.99958558, -1.70627019,  1.9507754}
                ,{-1.43411205,  1.50037656, -1.04855297, -1.42001794}
                ,{0.29484027, -0.79249401, -1.25279536,  0.77749036}});
        INDArray dataY = Nd4j.create(new int[]{1, -1, -1, 1, 1, -1}, new long[]{6}, DataType.INT32);
        
        /*INDArray column = dataX.dup();
        var tmp = Nd4j.sortWithIndices(column, 0, false)[0];
        int[][] indices = tmp.toIntMatrix();
        INDArray tmp2 = Nd4j.ones(6);
        tmp2.putScalar(new int[]{0,0}, -1);*/

        AdaBoostParallelClassifier clf = new AdaBoostParallelClassifier(150);
        var result = clf.fitIND(dataX, dataY, true);
        INDArray predY = clf.predictIND(dataX);
        assertArrayEquals(dataY.toIntVector(), predY.toIntVector(), "Wrong answer");
        log.info("Margin L1:{}", clf.getMarginL1IND(dataX));
    }
    
    @Test 
    void arraysCall() {
        double[][] dataX = new double[][]
                {{0.6476239, -0.81753611, -1.61389785, -0.21274028}
                ,{-2.37482060,  0.82768797, -0.38732682, -0.30230275}
                ,{1.51783379,  1.22140561, -0.51080514, -1.18063218}
                ,{-0.98740462,  0.99958558, -1.70627019,  1.9507754}
                ,{-1.43411205,  1.50037656, -1.04855297, -1.42001794}
                ,{0.29484027, -0.79249401, -1.25279536,  0.77749036}};
        int[] dataY = new int[]{1, -1, -1, 1, 1, -1};
        AdaBoostParallelClassifier clf = new AdaBoostParallelClassifier(150);
        var result = clf.fit(dataX, dataY);
        int[] predY = clf.predict(dataX);
        assertArrayEquals(dataY, predY, "Wrong answer");
        double marginL1 = clf.getMarginL1(dataX);
        assertEquals(0, clf.calculateMarginLoss(dataX, dataY, marginL1));
        log.info("Margin L1:{}", marginL1);
    }
}
