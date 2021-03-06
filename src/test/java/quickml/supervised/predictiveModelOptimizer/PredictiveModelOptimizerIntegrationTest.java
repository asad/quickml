package quickml.supervised.predictiveModelOptimizer;

import com.beust.jcommander.internal.Sets;
import com.google.common.collect.Maps;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import quickml.data.ClassifierInstance;
import quickml.data.OnespotDateTimeExtractor;
import quickml.supervised.classifier.Classifier;
import quickml.supervised.classifier.decisionTree.scorers.GiniImpurityScorer;
import quickml.supervised.classifier.decisionTree.scorers.InformationGainScorer;
import quickml.supervised.classifier.decisionTree.tree.attributeIgnoringStrategies.CompositeAttributeIgnoringStrategy;
import quickml.supervised.classifier.decisionTree.tree.attributeIgnoringStrategies.IgnoreAttributesInSet;
import quickml.supervised.classifier.decisionTree.tree.attributeIgnoringStrategies.IgnoreAttributesWithConstantProbability;
import quickml.supervised.classifier.randomForest.RandomForestBuilder;
import quickml.supervised.crossValidation.ClassifierLossChecker;
import quickml.supervised.crossValidation.data.OutOfTimeData;
import quickml.supervised.crossValidation.lossfunctions.WeightedAUCCrossValLossFunction;
import quickml.supervised.predictiveModelOptimizer.fieldValueRecommenders.FixedOrderRecommender;
import quickml.supervised.predictiveModelOptimizer.fieldValueRecommenders.MonotonicConvergenceRecommender;

import java.util.*;

import static java.util.Arrays.asList;
import static quickml.supervised.InstanceLoader.getAdvertisingInstances;
import static quickml.supervised.classifier.decisionTree.TreeBuilder.*;
import static quickml.supervised.classifier.randomForest.RandomForestBuilder.NUM_TREES;

public class PredictiveModelOptimizerIntegrationTest {


    private PredictiveModelOptimizer optimizer;

    @Before
    public void setUp() throws Exception {
        List<ClassifierInstance> advertisingInstances = getAdvertisingInstances();
        advertisingInstances = advertisingInstances.subList(0, 3000);
        optimizer = new PredictiveModelOptimizerBuilder<Classifier, ClassifierInstance>()
                .modelBuilder(new RandomForestBuilder<>())
                .dataCycler(new OutOfTimeData<>(advertisingInstances, 0.2, 12, new OnespotDateTimeExtractor()))
                .lossChecker(new ClassifierLossChecker<>(new WeightedAUCCrossValLossFunction(1.0)))
                .valuesToTest(createConfig())
                .iterations(3)
                .build();
    }

    @Ignore
    @Test
    public void testOptimizer() throws Exception {
        System.out.println("optimalConfig = " + optimizer.determineOptimalConfig());
    }
    
    private Map<String, FieldValueRecommender> createConfig() {
        Map<String, FieldValueRecommender> config = Maps.newHashMap();
        Set<String> attributesToIgnore = Sets.newHashSet();
        Set<String> exemptAttributes = Sets.newHashSet();
        exemptAttributes.addAll(Arrays.asList("seenClick", "seenCampaignClick", "seenPixel", "seenCampaignPixel", "seenCreativeClick", "seenCampaignClick"));

        attributesToIgnore.addAll(Arrays.asList("browser", "eap", "destinationId", "seenPixel", "internalCreativeId"));
        double probabilityOfDiscardingFromAttributesToIgnore = 0.3;
        CompositeAttributeIgnoringStrategy compositeAttributeIgnoringStrategy = new CompositeAttributeIgnoringStrategy(Arrays.asList(
                new IgnoreAttributesWithConstantProbability(0.7), new IgnoreAttributesInSet(attributesToIgnore, probabilityOfDiscardingFromAttributesToIgnore)
        ));
        config.put(ATTRIBUTE_IGNORING_STRATEGY, new FixedOrderRecommender(new IgnoreAttributesWithConstantProbability(0.7), compositeAttributeIgnoringStrategy ));
        config.put(NUM_TREES, new MonotonicConvergenceRecommender(asList(2, 4, 8), 0.02));
        config.put(MAX_DEPTH, new FixedOrderRecommender( 4, 8, 16));//Integer.MAX_VALUE, 2, 3, 5, 6, 9));
        config.put(MIN_SCORE, new FixedOrderRecommender(0.00000000000001));//, Double.MIN_VALUE, 0.0, 0.000001, 0.0001, 0.001, 0.01, 0.1));
        config.put(MIN_OCCURRENCES_OF_ATTRIBUTE_VALUE, new FixedOrderRecommender(2, 11, 16, 30 ));
        config.put(MIN_LEAF_INSTANCES, new FixedOrderRecommender(0, 20, 40));
        config.put(SCORER, new FixedOrderRecommender(new InformationGainScorer(), new GiniImpurityScorer()));
        config.put(DEGREE_OF_GAIN_RATIO_PENALTY, new FixedOrderRecommender(1.0, 0.75, .5 ));
        config.put(MIN_SPLIT_FRACTION, new FixedOrderRecommender(0.01, 0.25, .5 ));
        config.put(EXEMPT_ATTRIBUTES, new FixedOrderRecommender(exemptAttributes));

        return config;
    }



}
