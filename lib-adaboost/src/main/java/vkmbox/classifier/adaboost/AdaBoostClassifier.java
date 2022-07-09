package vkmbox.classifier.adaboost;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import vkmbox.classifier.util.RademacherUtil;
import vkmbox.classifier.base.DecisionStumpContinuous;

import java.util.Map;
import java.util.List;
import java.time.Clock;
import java.time.ZoneId;
import java.util.ArrayList;

/**
 *
 * @author vkmbox
 */
public abstract class AdaBoostClassifier {
    protected static final String METHOD_FIT_CALLED = "Method fit is called. Data: {} rows, {} columns";
    protected static final String METHOD_FIT_FINISHED = "Method fit is finished with result {}";

    protected static final int[] SIGNS = {-1, 1};
    protected static final Clock CLOCK = Clock.system(ZoneId.systemDefault());
    
    protected final double tolerance;
    protected final int defaultEstimators;
    
    public AdaBoostClassifier(int estimators, double tolerance) {
        this.tolerance = tolerance;
        this.defaultEstimators = estimators;
    }

    protected void log(boolean trace, Map<String, List> history
            , double minimalError, double[] weights, long timeStart) {
        if (trace) {
            history.computeIfAbsent("time", arg -> new ArrayList<Long>()).add(CLOCK.millis() - timeStart);
            history.computeIfAbsent("error", arg -> new ArrayList<Double>()).add(minimalError);
            history.computeIfAbsent("d_t", arg -> new ArrayList<double[]>()).add(weights);
        }
    }
    
    protected abstract FitResult internalFitIND
            (INDArray dataX, INDArray dataY, int estimators, boolean trace);
    
    public String fitIND(INDArray dataX, INDArray dataY) {
        var response = internalFitIND(dataX, dataY, defaultEstimators, false);
        return response.getResult().name();
    }
    
    public FitResult fitIND(INDArray dataX, INDArray dataY, boolean trace) {
        return internalFitIND(dataX, dataY, defaultEstimators, trace);
    }
    
    public String fit(double[][] dataX, int[] dataY) {
        return fit(dataX, dataY, defaultEstimators);
    }
    
    public String fit(double[][] dataX, int[] dataY, int estimators) {
        INDArray indY = Nd4j.create(dataY, new long[]{dataY.length}, DataType.INT32);
        var response = internalFitIND(Nd4j.create(dataX), indY, estimators, false);
        return response.getResult().name();
    }
            
    public abstract INDArray predictRaw(INDArray dataX);

    public INDArray predictIND(INDArray dataX) {
        return Transforms.sign(predictRaw(dataX));
    }
    
    public int[] predict(double[][] dataX) {
        return predictIND(Nd4j.create(dataX)).toIntVector();
    }
    
    /*
    * Following methods are proxy for static calls for the py4j convinience
    */
    public double calculateMarginLoss(double[][] dataX, int[] dataY, double rho) {
        INDArray indY = Nd4j.create(dataY, new long[]{dataY.length}, DataType.INT32);
        return calculateMarginLossIND(Nd4j.create(dataX), indY, rho);
    }
    
    public double calculateMarginLossIND(INDArray dataX, INDArray dataY, double rho) {
        int samplesCount = dataX.rows();
        INDArray indY = Nd4j.create(dataY.toDoubleVector());
        double[] buffer = indY.mul(predictRaw(dataX)).toDoubleVector();
        double sum = 0.0;
        for (double value: buffer) {
            if (value <= 0) {
                sum += 1.0;
            } else if (value <= rho) {
                sum += 1.0 - (value/rho);
            }
        }
        return sum/samplesCount;
    }
    
    public double getMarginL1(double[][] dataX) {
        return getMarginL1IND(Nd4j.create(dataX));
    }
    
    public double getMarginL1IND(INDArray dataX) {
        INDArray buffer = predictRaw(dataX);
        //double alphaModulo = ensembleAlphas.stream().mapToDouble(Double::doubleValue).sum() + tolerance;
        return buffer.aminNumber().doubleValue();//alphaModulo;
    }
    
    public double calculateRademacherForBiclassifiers(double[][] dataX, long subsetSize) {
        return calculateRademacherForBiclassifiersIND(Nd4j.create(dataX), subsetSize);
    }
    
    public double calculateRademacherForBiclassifiersIND(INDArray dataX, long subsetSize) {
        return RademacherUtil.calculateForBiclassifiers(dataX, subsetSize);
    }
    
    /*
    * Auxiliary classes
    */
    @Getter
    @RequiredArgsConstructor
    public static class StampResult {
        private final Double minimalError;
        private final DecisionStumpContinuous decisionStump;
        
        public static StampResult of(Double minimalError, DecisionStumpContinuous decisionStump) {
            return new StampResult(minimalError, decisionStump);
        }
    }
    
    public enum FitResultValue {
        ERROR_LEVEL_EXCEEDED, ERROR_FREE_CLASSIFIER_FOUND, ITERATIONS_EXCEEDED;
    }
    
    @Getter
    @RequiredArgsConstructor
    public static class FitResult {
        private final FitResultValue result;
        private final Map<String, List> history;
        
        public static FitResult of(FitResultValue result, Map<String, List> history) {
            return new FitResult(result, history);
        }
    }
}
