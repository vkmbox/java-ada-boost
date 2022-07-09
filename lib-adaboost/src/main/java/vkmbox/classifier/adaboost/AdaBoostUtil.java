package vkmbox.classifier.adaboost;

import static vkmbox.classifier.util.GeometryUtil.getNorm1;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import vkmbox.classifier.base.DecisionStumpContinuous;

import java.util.List;

public class AdaBoostUtil {
    public static INDArray predictRaw
            ( INDArray dataX, List<DecisionStumpContinuous> ensembleClassifiers
            , List<Double> ensembleAlphas, double tolerance
            ) {
        double alphaNorm1 = getNorm1(ensembleAlphas) + tolerance;
        int samplesCount = dataX.rows();
        INDArray buffer = Nd4j.zeros(new int[]{samplesCount});
        for (int index = 0; index < ensembleAlphas.size(); index++) {
            buffer.addi(ensembleClassifiers.get(index)
                .classifyIND(dataX, DataType.DOUBLE).mul(ensembleAlphas.get(index)));
        }
        return buffer.divi(alphaNorm1);
    }
}
