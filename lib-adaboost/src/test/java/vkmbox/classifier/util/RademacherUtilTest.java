package vkmbox.classifier.util;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;

public class RademacherUtilTest {
    
    @Test 
    void rademacherVectorTest() {
        INDArray vectorsH = Nd4j.create(new double[][] {{1,1}, {1,2}});
        assertEquals(1.5, RademacherUtil.calculateMaxOnRademacherVector(vectorsH, Nd4j.create(new double[]{1.0,1.0})));
        assertEquals(0, RademacherUtil.calculateMaxOnRademacherVector(vectorsH, Nd4j.create(new double[]{1.0,-1.0})));
        assertEquals(0.5, RademacherUtil.calculateMaxOnRademacherVector(vectorsH, Nd4j.create(new double[]{-1.0,1.0})));
        assertEquals(-1, RademacherUtil.calculateMaxOnRademacherVector(vectorsH, Nd4j.create(new double[]{-1.0,-1.0})));
    }
    
    @Test 
    void rademacherBiclassifierTest() {
        INDArray dataX = Nd4j.create(new double[][]
                {{0.6476239, -0.81753611, -1.61389785, -0.21274028}
                ,{-2.37482060,  0.82768797, -0.38732682, -0.30230275}
                ,{1.51783379,  1.22140561, -0.51080514, -1.18063218}
                ,{-0.98740462,  0.99958558, -1.70627019,  1.9507754}
                ,{-1.43411205,  1.50037656, -1.04855297, -1.42001794}
                ,{0.29484027, -0.79249401, -1.25279536,  0.77749036}});
        
        assertEquals(0.8333333333333329, RademacherUtil.calculateForBiclassifiers(dataX, 64));
    }
}
