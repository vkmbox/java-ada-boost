package vkmbox.classifier.adaboost;

import static org.nd4j.common.util.MathUtils.sum;
import static vkmbox.classifier.adaboost.AdaBoostClassifier.FitResultValue.ITERATIONS_EXCEEDED;
import static vkmbox.classifier.adaboost.AdaBoostClassifier.FitResultValue.ERROR_LEVEL_EXCEEDED;
import static vkmbox.classifier.adaboost.AdaBoostClassifier.FitResultValue.ERROR_FREE_CLASSIFIER_FOUND;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import vkmbox.classifier.base.DecisionStumpContinuous;

import java.util.Map;
import java.util.List;
import java.util.Arrays;
import java.util.HashMap;
import java.util.ArrayList;

@Slf4j
public class AdaBoostStandardClassifier extends AdaBoostClassifier {
    private final List<Double> ensembleAlphas = new ArrayList<>();
    private final List<DecisionStumpContinuous> ensembleClassifiers = new ArrayList<>();
    
    public AdaBoostStandardClassifier(int estimators, double tolerance) {
        super(estimators, tolerance);
    }

    public AdaBoostStandardClassifier(int estimators) {
        super(estimators, 1e-10);
    }
    
    @Override
    protected FitResult internalFitIND
            (INDArray dataX, INDArray dataY, int estimators, boolean trace) {
        log.info(METHOD_FIT_CALLED, dataX.rows(), dataX.columns());
        ensembleAlphas.clear();
        ensembleClassifiers.clear();

        long timeStart = CLOCK.millis();
        int samplesCount = dataX.rows();
        double[] weights = new double[samplesCount];
        Arrays.fill(weights, 1.0/samplesCount);
        Map<String, List> history = trace? new HashMap<>(): null;

        for (int dummy = 0; dummy < estimators; dummy++) {
            var optimum = getDecisionStump(dataX, dataY, weights);
            if (optimum.getMinimalError() >= 0.5) {
                log.warn(METHOD_FIT_FINISHED, ERROR_LEVEL_EXCEEDED);
                return FitResult.of(ERROR_LEVEL_EXCEEDED, history);
            }
            double alphaT = optimum.getMinimalError() == 0 ? 1
                    : 0.5 * Math.log((1 - optimum.getMinimalError())/(optimum.getMinimalError()+tolerance));
            ensembleAlphas.add(alphaT);
            ensembleClassifiers.add(optimum.getDecisionStump());
            log(trace, history, optimum.getMinimalError(), weights, timeStart);
            if (optimum.getMinimalError() == 0) {
                log.info(METHOD_FIT_FINISHED, ERROR_FREE_CLASSIFIER_FOUND);
                return FitResult.of(ERROR_FREE_CLASSIFIER_FOUND, history);
            }
            int[] forecast = optimum.getDecisionStump().classify(dataX);
            for (int sampleNumber = 0; sampleNumber < samplesCount; sampleNumber++) {
                weights[sampleNumber] *= Math.exp(-alphaT * dataY.getInt(sampleNumber) * forecast[sampleNumber]);
            }
            double weightSum = sum(weights);
            for (int sampleNumber = 0; sampleNumber < samplesCount; sampleNumber++) {
                weights[sampleNumber] = weights[sampleNumber]/(weightSum+tolerance);
            }
        }
        log.info(METHOD_FIT_FINISHED, ITERATIONS_EXCEEDED);
        return FitResult.of(ITERATIONS_EXCEEDED, history);
    }
    
    @Override
    public INDArray predictRaw(INDArray dataX) {
        return AdaBoostUtil.predictRaw
            (dataX, ensembleClassifiers, ensembleAlphas, tolerance);
    }
    
    private StampResult getDecisionStump(INDArray dataX, INDArray dataY, double[] weights) {
        int samplesCount = dataX.rows(), featuresCount = dataX.columns();
        double minimalError = Double.MAX_VALUE, minimalThreshold = Double.MAX_VALUE;
        int minimalFeature = 0, minimalSign = 0;

        for (int featureNumber = 0; featureNumber < featuresCount; featureNumber++) {
            INDArray feature = dataX.getColumn(featureNumber);
            for (int sampleNumber = 0; sampleNumber < samplesCount; sampleNumber++) {
                double threshold = feature.getDouble(sampleNumber);
                for (int sign : SIGNS) {
                    double currentError = DecisionStumpContinuous.getError
                        (dataX, dataY, featureNumber, weights, sign, threshold);
                    if (minimalError > currentError) {
                        minimalError = currentError;
                        minimalFeature = featureNumber;
                        minimalSign = sign; 
                        minimalThreshold = threshold;
                    }
                }
            }
        }
        return StampResult.of(minimalError
                , new DecisionStumpContinuous(minimalFeature, minimalSign, minimalThreshold));
    }
}
