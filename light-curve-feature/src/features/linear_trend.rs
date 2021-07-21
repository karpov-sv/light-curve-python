use crate::evaluator::*;
use crate::fit::fit_straight_line;

/// The slope and noise of the light curve without observation errors in the linear fit
///
/// Least squares fit of the linear stochastic model with constant Gaussian noise $\Sigma$ assuming
/// observation errors to be zero:
/// $$
/// m_i = c + \mathrm{slope}\\,t_i + \Sigma \varepsilon_i,
/// $$
/// where $c$ and $\Sigma$ are constants,
/// $\\{\varepsilon_i\\}$ are standard distributed random variables.
/// $\mathrm{slope}$ and $\sigma_\mathrm{slope}$ are returned, if $N = 2$ than no least squares fit is done, a
/// slope between a pair of observations $(m_1 - m_0) / (t_1 - t_0)$ and $0$ are returned.
///
/// - Depends on: **time**, **magnitude**
/// - Minimum number of observations: **2**
/// - Number of features: **2**
#[derive(Clone, Default, Debug, Serialize)]
pub struct LinearTrend {}

impl LinearTrend {
    pub fn new() -> Self {
        Self {}
    }
}

lazy_info!(
    LINEAR_TREND_INFO,
    size: 2,
    min_ts_length: 2,
    t_required: true,
    m_required: true,
    w_required: false,
    sorting_required: true,
);

impl<T> FeatureEvaluator<T> for LinearTrend
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        let size = self.check_ts_length(ts)?;
        if size == 2 {
            return Ok(vec![
                (ts.m.sample[1] - ts.m.sample[0]) / (ts.t.sample[1] - ts.t.sample[0]),
                T::zero(),
            ]);
        }
        let result = fit_straight_line(ts, false);
        Ok(vec![result.slope, T::sqrt(result.slope_sigma2)])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &LINEAR_TREND_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["linear_trend", "linear_trend_sigma"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![
            "linear trend without respect to observation errors",
            "error of slope of linear fit without respect to observation errors",
        ]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(linear_trend_info, LinearTrend::default());

    feature_test!(
        linear_trend,
        [LinearTrend::new()],
        [1.38198758, 0.24532195657979344],
        [1.0_f32, 3.0, 5.0, 7.0, 11.0, 13.0],
        [1.0_f32, 2.0, 3.0, 8.0, 10.0, 19.0],
    );

    /// See [Issue #3](https://github.com/hombit/light-curve/issues/3)
    #[test]
    fn linear_trend_finite_sigma() {
        let eval = LinearTrend::default();
        let x = [
            58216.51171875,
            58217.48828125,
            58217.5078125,
            58218.49609375,
            58218.5078125,
            58228.45703125,
            58229.45703125,
            58230.47265625,
            58242.46484375,
            58244.4765625,
            58244.49609375,
            58246.4453125,
            58247.46484375,
            58249.45703125,
            58249.47265625,
            58254.46484375,
            58255.4453125,
            58256.4375,
            58256.46484375,
            58257.421875,
            58257.45703125,
            58258.4453125,
            58259.44921875,
            58261.48828125,
            58262.38671875,
            58263.421875,
            58266.359375,
            58268.42578125,
            58269.4140625,
            58270.44140625,
            58271.4609375,
            58273.421875,
            58274.33984375,
            58275.40234375,
            58276.4296875,
            58277.39453125,
            58279.40625,
            58280.375,
            58282.4453125,
            58283.38671875,
            58285.3828125,
            58286.34375,
            58288.44921875,
            58289.4453125,
            58290.3828125,
            58291.484375,
            58292.46875,
            58293.3203125,
            58294.46875,
            58296.32421875,
            58297.4140625,
            58298.43359375,
            58299.40234375,
            58300.37890625,
            58301.3828125,
            58303.37890625,
            58304.3203125,
            58305.3828125,
            58307.3828125,
            58310.38671875,
            58311.3828125,
            58312.421875,
            58313.38671875,
            58314.41796875,
            58316.33984375,
            58317.40625,
            58318.33984375,
            58320.36328125,
            58321.3515625,
            58322.39453125,
            58323.2734375,
            58324.23828125,
            58325.29296875,
            58326.33984375,
            58327.33984375,
            58329.34375,
            58330.37890625,
            58332.32421875,
            58333.32421875,
            58334.35546875,
            58336.34375,
            58338.3359375,
            58340.3046875,
            58341.328125,
            58342.33203125,
            58343.32421875,
            58344.30859375,
            58345.32421875,
            58346.31640625,
            58349.30078125,
            58351.21484375,
            58354.2578125,
            58354.359375,
            58355.28125,
            58356.1953125,
            58356.29296875,
            58357.2109375,
            58358.25390625,
            58360.27734375,
            58366.1875,
            58370.2578125,
            58373.171875,
            58374.171875,
            58425.0859375,
            58427.109375,
            58428.1015625,
            58431.1328125,
        ];
        let y = [
            18.614999771118164,
            18.714000701904297,
            18.665000915527344,
            18.732999801635742,
            18.658000946044922,
            18.70199966430664,
            18.641000747680664,
            18.631999969482422,
            18.659000396728516,
            18.68899917602539,
            18.75,
            18.767000198364258,
            18.70400047302246,
            18.85300064086914,
            18.7450008392334,
            18.770000457763672,
            18.67799949645996,
            18.70800018310547,
            18.724000930786133,
            18.70400047302246,
            18.680999755859375,
            18.733999252319336,
            18.64900016784668,
            18.67099952697754,
            18.707000732421875,
            18.781999588012695,
            18.691999435424805,
            18.695999145507812,
            18.684999465942383,
            18.72800064086914,
            18.68600082397461,
            18.743000030517578,
            18.718000411987305,
            18.645000457763672,
            18.708999633789062,
            18.69700050354004,
            18.704999923706055,
            18.71500015258789,
            18.729000091552734,
            18.69499969482422,
            18.660999298095703,
            18.718000411987305,
            18.628000259399414,
            18.76799964904785,
            18.733999252319336,
            18.735000610351562,
            18.70800018310547,
            18.753999710083008,
            18.66699981689453,
            18.735000610351562,
            18.697999954223633,
            19.034000396728516,
            18.628999710083008,
            18.711000442504883,
            18.76799964904785,
            18.701000213623047,
            18.687000274658203,
            18.733999252319336,
            18.715999603271484,
            18.69099998474121,
            18.711999893188477,
            18.715999603271484,
            18.764999389648438,
            18.663999557495117,
            18.722000122070312,
            18.70400047302246,
            18.690000534057617,
            18.67099952697754,
            18.65999984741211,
            18.7549991607666,
            18.666000366210938,
            18.60700035095215,
            18.715999603271484,
            18.732999801635742,
            18.788999557495117,
            18.791000366210938,
            18.714000701904297,
            18.738000869750977,
            18.672000885009766,
            18.74799919128418,
            18.69099998474121,
            18.718000411987305,
            18.64699935913086,
            18.70800018310547,
            18.656999588012695,
            18.672000885009766,
            18.711999893188477,
            18.781999588012695,
            18.628000259399414,
            18.698999404907227,
            18.722000122070312,
            18.70599937438965,
            18.645000457763672,
            18.80500030517578,
            18.820999145507812,
            18.75,
            18.77400016784668,
            18.761999130249023,
            19.656999588012695,
            18.76300048828125,
            18.71299934387207,
            18.750999450683594,
            18.70800018310547,
            18.71500015258789,
            18.638999938964844,
            18.677000045776367,
            18.69700050354004,
        ];
        let mut ts = TimeSeries::new_without_weight(&x, &y);
        let sigma: f32 = eval.eval(&mut ts).unwrap()[1];
        assert!(sigma.is_finite());
    }

    /// See [Issue #3](https://github.com/hombit/light-curve/issues/3)
    #[test]
    fn linear_trend_finite_trend_and_sigma_1() {
        let eval = LinearTrend::default();
        let x = [
            58231.140625,
            58303.4765625,
            58314.44140625,
            58315.484375,
            58316.46875,
            58319.47265625,
            58321.48828125,
            58323.48828125,
            58324.48828125,
            58325.484375,
            58329.48828125,
            58330.41796875,
            58333.48828125,
            58334.4453125,
            58335.46484375,
            58336.4609375,
            58337.48828125,
            58338.48828125,
            58342.484375,
            58343.484375,
            58344.46484375,
            58345.47265625,
            58346.44140625,
            58347.44921875,
            58348.4453125,
            58349.484375,
            58350.4921875,
            58351.484375,
            58352.48828125,
            58353.453125,
            58353.49609375,
            58354.453125,
            58354.484375,
            58355.40625,
            58355.48046875,
            58356.453125,
            58356.484375,
            58357.44921875,
            58357.5078125,
            58358.44921875,
            58359.48828125,
            58360.49609375,
            58361.5078125,
            58363.47265625,
            58364.4921875,
            58365.48828125,
            58366.484375,
            58367.4921875,
            58368.46484375,
            58369.4296875,
            58370.48828125,
            58371.45703125,
            58372.4921875,
            58373.4921875,
            58374.48828125,
            58375.4921875,
            58376.4375,
            58377.4453125,
            58378.42578125,
            58379.4296875,
            58380.453125,
            58382.5,
            58383.515625,
            58384.51171875,
            58385.5078125,
            58386.4375,
            58387.46484375,
            58388.52734375,
            58389.48828125,
            58397.42578125,
            58424.35546875,
            58425.33203125,
            58426.41796875,
            58427.44921875,
            58430.45703125,
            58431.28515625,
            58432.28515625,
            58434.34765625,
            58436.33984375,
            58437.34765625,
            58441.41015625,
            58443.38671875,
            58447.41015625,
            58449.36328125,
            58450.35546875,
            58455.2890625,
            58455.36328125,
            58456.22265625,
            58456.27734375,
            58457.26953125,
            58464.265625,
            58465.265625,
            58468.27734375,
            58471.2421875,
            58472.265625,
            58474.3203125,
            58476.3046875,
            58480.31640625,
            58481.31640625,
            58482.19921875,
        ];
        let y = [
            19.08300018310547,
            18.988000869750977,
            19.086999893188477,
            18.95400047302246,
            19.076000213623047,
            19.076000213623047,
            19.090999603271484,
            18.966999053955078,
            19.041000366210938,
            19.089000701904297,
            19.05699920654297,
            19.097000122070312,
            19.132999420166016,
            19.104000091552734,
            19.06100082397461,
            19.128000259399414,
            19.099000930786133,
            19.06599998474121,
            19.100000381469727,
            19.08300018310547,
            19.1200008392334,
            19.115999221801758,
            19.128999710083008,
            19.07900047302246,
            19.16699981689453,
            19.179000854492188,
            19.1560001373291,
            19.16200065612793,
            19.110000610351562,
            19.14900016784668,
            19.10700035095215,
            19.104999542236328,
            19.145000457763672,
            19.091999053955078,
            19.091999053955078,
            19.225000381469727,
            19.086000442504883,
            19.054000854492188,
            19.17799949645996,
            19.17099952697754,
            19.1200008392334,
            19.02899932861328,
            19.18000030517578,
            19.10700035095215,
            19.118000030517578,
            19.128000259399414,
            19.166000366210938,
            19.08300018310547,
            19.124000549316406,
            19.106000900268555,
            19.10700035095215,
            19.097999572753906,
            19.106000900268555,
            19.107999801635742,
            19.075000762939453,
            18.965999603271484,
            19.134000778198242,
            19.136999130249023,
            19.150999069213867,
            19.1200008392334,
            19.149999618530273,
            19.152999877929688,
            19.013999938964844,
            19.06800079345703,
            19.101999282836914,
            19.093000411987305,
            19.107999801635742,
            19.054000854492188,
            19.062000274658203,
            19.174999237060547,
            19.05299949645996,
            19.04400062561035,
            19.149999618530273,
            19.136999130249023,
            19.152999877929688,
            19.16900062561035,
            18.986000061035156,
            19.204999923706055,
            19.091999053955078,
            19.038999557495117,
            19.246999740600586,
            19.107999801635742,
            19.082000732421875,
            19.148000717163086,
            19.128999710083008,
            19.1560001373291,
            19.187999725341797,
            19.17300033569336,
            19.163000106811523,
            19.1299991607666,
            19.158000946044922,
            19.163999557495117,
            19.10099983215332,
            19.125,
            19.138999938964844,
            19.09000015258789,
            19.19300079345703,
            19.128000259399414,
            19.143999099731445,
            19.21500015258789,
        ];
        let mut ts: TimeSeries<f32> = TimeSeries::new_without_weight(&x, &y);
        let actual = eval.eval(&mut ts).unwrap();
        assert!(actual.iter().all(|x| x.is_finite()));
    }

    /// See [Issue #3](https://github.com/hombit/light-curve/issues/3)
    #[test]
    fn linear_trend_finite_trend_and_sigma_2() {
        let eval = LinearTrend::default();
        let x = [
            58231.140625,
            58303.4765625,
            58314.44140625,
            58315.484375,
            58316.46875,
            58319.47265625,
            58321.48828125,
            58323.48828125,
            58324.48828125,
            58325.484375,
            58329.48828125,
            58330.41796875,
            58333.48828125,
            58334.4453125,
            58335.46484375,
            58336.4609375,
            58337.48828125,
            58338.48828125,
            58342.484375,
            58343.484375,
            58344.46484375,
            58345.47265625,
            58346.44140625,
            58347.44921875,
            58348.4453125,
            58349.484375,
            58350.4921875,
            58351.484375,
            58352.48828125,
            58353.453125,
            58353.49609375,
            58354.453125,
            58354.484375,
            58355.40625,
            58355.48046875,
            58356.453125,
            58356.484375,
            58357.44921875,
            58357.5078125,
            58358.44921875,
            58359.48828125,
            58360.49609375,
            58361.5078125,
            58363.47265625,
            58364.4921875,
            58365.48828125,
            58366.484375,
            58367.4921875,
            58368.46484375,
            58369.4296875,
            58370.48828125,
            58371.45703125,
            58372.4921875,
            58373.4921875,
            58374.48828125,
            58375.4921875,
            58376.4375,
            58377.4453125,
            58378.42578125,
            58379.4296875,
            58380.453125,
            58382.5,
            58383.515625,
            58384.51171875,
            58385.5078125,
            58386.4375,
            58387.46484375,
            58388.52734375,
            58389.48828125,
            58397.42578125,
            58424.35546875,
            58425.33203125,
            58426.41796875,
            58427.44921875,
            58430.45703125,
            58431.28515625,
            58432.28515625,
            58434.34765625,
            58436.33984375,
            58437.34765625,
            58441.41015625,
            58443.38671875,
            58447.41015625,
            58449.36328125,
            58450.35546875,
            58455.2890625,
            58455.36328125,
            58456.22265625,
            58456.27734375,
            58457.26953125,
            58464.265625,
            58465.265625,
            58468.27734375,
            58471.2421875,
            58472.265625,
            58474.3203125,
            58476.3046875,
            58480.31640625,
            58481.31640625,
            58482.19921875,
        ];
        let y = [
            17.996000289916992,
            18.047000885009766,
            17.983999252319336,
            18.006999969482422,
            18.062000274658203,
            18.02899932861328,
            18.003999710083008,
            17.97599983215332,
            17.992000579833984,
            18.011999130249023,
            18.055999755859375,
            18.013999938964844,
            17.979999542236328,
            18.023000717163086,
            18.034000396728516,
            18.024999618530273,
            18.027999877929688,
            18.017000198364258,
            18.01300048828125,
            18.040000915527344,
            18.006999969482422,
            18.016000747680664,
            18.006999969482422,
            18.000999450683594,
            17.99799919128418,
            18.000999450683594,
            18.038999557495117,
            18.047000885009766,
            18.011999130249023,
            18.03700065612793,
            18.027999877929688,
            18.0,
            18.006000518798828,
            17.957000732421875,
            18.013999938964844,
            18.017000198364258,
            18.04199981689453,
            18.01799964904785,
            18.101999282836914,
            18.051000595092773,
            18.05699920654297,
            18.01300048828125,
            18.027000427246094,
            18.027000427246094,
            18.031999588012695,
            18.0049991607666,
            18.009000778198242,
            18.059999465942383,
            18.018999099731445,
            18.024999618530273,
            18.035999298095703,
            18.02400016784668,
            18.038000106811523,
            18.06100082397461,
            18.02899932861328,
            18.038000106811523,
            18.047000885009766,
            18.01799964904785,
            18.0310001373291,
            18.034000396728516,
            17.97100067138672,
            18.02400016784668,
            18.033000946044922,
            18.018999099731445,
            18.05500030517578,
            18.030000686645508,
            18.02199935913086,
            18.014999389648438,
            18.006000518798828,
            18.045000076293945,
            17.981000900268555,
            18.040000915527344,
            18.003000259399414,
            18.02199935913086,
            18.04199981689453,
            18.04800033569336,
            18.045000076293945,
            18.059999465942383,
            18.062000274658203,
            18.058000564575195,
            18.0310001373291,
            18.041000366210938,
            18.20599937438965,
            17.993000030517578,
            18.030000686645508,
            17.996000289916992,
            18.06599998474121,
            18.030000686645508,
            18.05900001525879,
            18.024999618530273,
            18.05500030517578,
            17.98900032043457,
            18.017000198364258,
            17.950000762939453,
            17.996999740600586,
            18.03499984741211,
            17.98900032043457,
            17.986000061035156,
            18.020999908447266,
            18.075000762939453,
        ];
        let mut ts: TimeSeries<f32> = TimeSeries::new_without_weight(&x, &y);
        let actual = eval.eval(&mut ts).unwrap();
        assert!(actual.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn linear_trend_finite_trend_and_sigma_3() {
        let eval = LinearTrend::default();
        let x = [
            198.39394, 198.40166, 198.43057, 198.45149, 198.45248, 198.4768, 198.48457, 198.48549,
            216.39883, 216.39975, 217.3903, 217.41743, 217.4417, 217.46191, 218.34486, 218.3973,
            218.43736, 218.4782, 218.5021, 219.3902, 219.4144, 219.43823, 219.45737, 219.47935,
            219.48029, 219.50168, 222.37775, 222.39838, 224.43896, 226.36752, 226.38454, 226.40622,
            226.40714, 226.43475, 226.46725, 226.46819, 226.4971, 227.40437, 229.38954, 229.41095,
            229.41333, 229.4361, 230.43388, 230.44582, 231.3048, 231.36673, 232.3927, 243.30058,
            243.32713, 244.3279, 244.36636, 246.42494, 247.2857, 247.40685, 247.4248, 249.44841,
            252.38995, 252.39087, 254.37566, 255.28339, 255.30199, 257.30103, 257.32593, 257.3523,
            257.37186, 257.38855, 257.406, 258.40897, 258.4266, 263.36945, 263.39005, 263.40744,
            266.29138, 266.29904, 266.32605, 266.32986, 268.38016, 269.34402, 269.36942, 269.37033,
            269.39255, 270.3333, 270.3504, 270.35907, 271.3887, 272.28064, 272.3053, 272.30624,
            272.32397, 273.32425, 273.34964, 273.37308, 274.30508, 274.32437, 274.34604, 276.29034,
            276.3069, 276.32434, 277.28595, 277.3864, 277.40887, 278.26627, 278.285, 278.28592,
            278.3105, 279.25125, 279.26828, 279.2863, 280.30655, 280.32486, 280.34262, 281.2201,
            281.32944, 281.3454, 281.34686, 282.30692, 282.3234, 282.34473, 283.32944, 283.34775,
            283.36725, 284.2814, 284.30304, 284.30444, 284.31894, 285.30908, 285.32278, 285.39072,
            286.30627, 286.32785, 287.26617, 287.2856, 287.28653, 287.30597, 288.27066, 288.28412,
            288.30637, 289.23962, 289.25977, 290.26422, 290.33035, 290.33127, 290.34775, 291.26752,
            291.28244, 291.30322, 292.24774, 292.3009, 292.32034, 293.24948, 293.2504, 293.2614,
            293.28522, 294.2839, 294.30447, 295.26587, 295.28528, 295.30753, 296.24985, 296.28497,
            297.26114, 297.2795, 297.30798, 298.26505, 298.28934, 298.2994, 299.34207, 299.3439,
            299.36365, 299.38684, 300.32657, 300.34445, 300.3585, 301.1824, 301.3463, 301.36996,
            302.3121, 302.32617, 302.34195, 302.34286, 303.30817, 303.32306, 303.34805, 304.32675,
            304.34723, 305.18744, 305.34937, 305.36835, 306.18637, 306.199, 306.3686, 307.18417,
            307.20242, 307.3693, 308.24564, 308.24655, 308.31927, 311.1842, 311.20007, 312.2234,
            316.24084, 319.22058, 319.2354, 319.24335, 319.28165, 320.2651, 321.22864, 321.24417,
            321.2779, 322.23578, 322.23996, 322.263, 322.28027, 323.24234, 324.1836, 324.19937,
            324.22025, 325.2254, 325.22632, 325.23944, 325.26257, 326.27972, 326.29578, 327.28586,
            327.30435, 327.3241, 328.2859, 328.28778, 329.30374, 329.32187, 330.30502, 330.32248,
            330.33926, 331.17816, 331.19775, 331.1996, 331.21732, 332.25946, 334.16504, 337.19443,
            340.17987, 343.15842, 346.17615, 349.24045, 349.2414, 349.26685, 349.27853, 350.16284,
            350.18295, 350.198, 351.1598, 351.17282, 351.2005, 352.21957, 352.24164, 352.24255,
            352.26224, 353.1589, 353.1766, 353.19882, 353.24045, 353.28317, 353.29752, 354.15997,
            354.19803, 354.2399, 355.23972, 355.2543, 356.2045, 356.22006, 356.2378, 356.24338,
            356.2596, 357.1807, 357.23642, 357.25885, 357.28036, 358.19315, 358.21973, 358.2416,
            359.21567, 359.26645, 359.28586, 360.2256, 360.2449, 360.28036, 361.2598, 361.28598,
            362.15714, 364.14243, 364.15277, 364.17447, 365.1356, 365.1559, 365.17996, 365.18088,
            367.15826, 367.1771, 367.20325, 368.1365, 368.1566, 368.15842, 368.18253, 369.1339,
            369.1569, 369.17902, 370.14294, 370.15485, 370.2592, 371.13763, 371.1583, 371.1592,
            371.17938, 372.1418, 372.15417, 372.18033, 373.12784, 373.15872, 373.18106, 374.13278,
            374.15768, 374.1586, 375.17548, 375.1987, 376.1628, 376.17404, 376.19482, 377.1364,
            377.13733, 377.16132, 377.17914, 378.136, 378.21686, 378.24747, 379.2173, 379.23883,
            381.1212, 381.1392, 381.14154, 381.15915, 382.14227, 382.15585, 383.1172, 383.1348,
            383.15805, 384.1447, 384.15216, 384.17374, 385.19153, 385.21136, 386.2009, 386.21588,
            387.1196, 387.1205, 387.1377, 387.15375, 389.2162, 390.20016, 390.2159, 390.21683,
            423.07803, 476.53094,
        ];
        let y = [
            16.591, 16.608, 16.615, 16.605, 16.601, 16.602, 16.608, 16.583, 16.618, 16.613, 16.619,
            16.611, 16.595, 16.581, 16.603, 16.577, 16.626, 16.586, 16.618, 16.596, 16.598, 16.576,
            16.583, 16.596, 16.604, 16.584, 16.616, 16.594, 16.584, 16.603, 16.602, 16.573, 16.625,
            16.61, 16.58, 16.594, 16.622, 16.583, 16.567, 16.636, 16.586, 16.602, 16.563, 16.587,
            16.563, 16.582, 16.602, 16.618, 16.594, 16.559, 16.613, 16.625, 16.609, 16.61, 16.593,
            16.61, 16.598, 16.591, 16.601, 16.609, 16.618, 16.587, 16.605, 16.586, 16.6, 16.59,
            16.621, 16.577, 16.611, 16.61, 16.599, 16.578, 16.581, 16.604, 16.565, 16.599, 16.611,
            16.605, 16.603, 16.608, 16.602, 16.602, 16.609, 16.583, 16.606, 16.6, 16.609, 16.61,
            16.587, 16.59, 16.604, 16.599, 16.591, 16.607, 16.599, 16.575, 16.588, 16.6, 16.59,
            16.594, 16.615, 16.592, 16.595, 16.616, 16.591, 16.598, 16.585, 16.611, 16.614, 16.606,
            16.621, 16.607, 16.594, 16.605, 16.611, 16.608, 16.621, 16.578, 16.609, 16.612, 16.619,
            16.616, 16.597, 16.61, 16.623, 16.613, 16.608, 16.6, 16.607, 16.573, 16.598, 16.603,
            16.609, 16.583, 16.601, 16.621, 16.601, 16.629, 16.607, 16.563, 16.604, 16.587, 16.584,
            16.587, 16.578, 16.595, 16.581, 16.591, 16.608, 16.583, 16.592, 16.611, 16.597, 16.575,
            16.615, 16.582, 16.59, 16.592, 16.607, 16.617, 16.626, 16.575, 16.579, 16.613, 16.592,
            16.584, 16.599, 16.606, 16.574, 16.601, 16.597, 16.612, 16.608, 16.605, 16.611, 16.596,
            16.626, 16.625, 16.573, 16.609, 16.592, 16.598, 16.603, 16.599, 16.615, 16.588, 16.623,
            16.603, 16.614, 16.576, 16.587, 16.608, 16.597, 16.595, 16.585, 16.624, 16.616, 16.584,
            16.619, 16.596, 16.605, 16.595, 16.616, 16.589, 16.591, 16.618, 16.589, 16.59, 16.6,
            16.6, 16.618, 16.578, 16.589, 16.582, 16.59, 16.578, 16.605, 16.583, 16.574, 16.596,
            16.577, 16.61, 16.6, 16.579, 16.538, 16.584, 16.596, 16.609, 16.58, 16.591, 16.614,
            16.612, 16.6, 16.611, 16.579, 16.556, 16.583, 16.59, 16.583, 16.586, 16.595, 16.597,
            16.579, 16.578, 16.555, 16.577, 16.59, 16.577, 16.58, 16.593, 16.576, 16.581, 16.591,
            16.595, 16.582, 16.604, 16.601, 16.607, 16.605, 16.604, 16.596, 16.596, 16.606, 16.601,
            16.596, 16.608, 16.61, 16.604, 16.575, 16.593, 16.602, 16.596, 16.61, 16.609, 16.604,
            16.601, 16.596, 16.566, 16.605, 16.591, 16.657, 16.564, 16.577, 16.601, 16.594, 16.602,
            16.608, 16.621, 16.588, 16.585, 16.607, 16.598, 16.594, 16.611, 16.602, 16.621, 16.581,
            16.62, 16.584, 16.601, 16.586, 16.573, 16.588, 16.58, 16.586, 16.576, 16.613, 16.605,
            16.605, 16.586, 16.602, 16.593, 16.575, 16.593, 16.591, 16.579, 16.593, 16.59, 16.601,
            16.581, 16.599, 16.599, 16.611, 16.62, 16.6, 16.588, 16.583, 16.588, 16.6, 16.601,
            16.614, 16.575, 16.602, 16.617, 16.608, 16.588, 16.6, 16.588, 16.587, 16.587, 16.6,
            16.614, 16.605, 16.623, 16.603, 16.604, 16.618, 16.592, 16.578, 16.59, 16.598, 16.572,
            16.609, 16.592, 16.574, 16.562, 16.558, 16.581, 16.581, 16.602, 16.581, 16.595,
        ];
        let mut ts: TimeSeries<f32> = TimeSeries::new_without_weight(&x, &y);
        let actual = eval.eval(&mut ts).unwrap();
        assert!(actual.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn linear_trend_finite_trend_and_sigma_4() {
        let eval = LinearTrend::default();
        let x = [
            198.39395, 198.40167, 198.4306, 198.4515, 198.45251, 198.47682, 198.4846, 198.4855,
            216.39883, 216.39977, 217.3903, 217.41743, 217.4417, 217.46191, 218.34488, 218.3973,
            218.43738, 218.47821, 218.50212, 219.39021, 219.41441, 219.43825, 219.45738, 219.47937,
            219.48029, 219.5017, 222.37775, 222.39839, 224.43896, 226.36752, 226.38455, 226.40623,
            226.40714, 226.43475, 226.46727, 226.46819, 226.49712, 227.40439, 229.38954, 229.41095,
            229.41335, 229.4361, 230.43388, 230.44582, 231.3048, 231.36674, 232.39272, 243.30058,
            243.32713, 244.3279, 244.36636, 246.42494, 247.2857, 247.40685, 247.4248, 249.44841,
            252.38995, 252.39085, 254.37566, 255.28339, 255.30199, 257.30103, 257.32593, 257.3523,
            257.37186, 257.38855, 257.406, 258.385, 258.40897, 258.4266, 262.3614, 263.39005,
            266.29138, 266.29904, 266.32605, 266.32986, 268.38016, 269.34402, 269.36942, 269.37033,
            269.39252, 270.3333, 270.3504, 270.35907, 271.38867, 272.28064, 272.3053, 272.3062,
            272.32397, 273.32425, 273.34964, 273.37305, 274.30508, 274.32437, 274.346, 276.29034,
            276.30685, 276.32434, 277.28595, 277.3864, 277.40887, 278.26627, 278.28497, 278.2859,
            278.3105, 279.25122, 279.26828, 279.2863, 280.30655, 280.32483, 280.34262, 281.2201,
            281.32944, 281.3454, 281.34683, 282.3069, 282.3234, 282.3447, 283.32944, 283.34772,
            283.36722, 284.2814, 284.303, 284.3044, 284.3189, 285.30908, 285.32278, 285.39072,
            286.30624, 286.32785, 287.26617, 287.2856, 287.28653, 287.30594, 288.27066, 288.2841,
            288.30634, 289.2396, 289.25977, 290.26422, 290.33035, 290.33124, 290.34775, 291.2675,
            291.2824, 291.30322, 292.2477, 292.30087, 292.32034, 293.24945, 293.25037, 293.26138,
            293.28522, 294.28387, 294.30447, 295.26587, 295.28525, 295.3075, 296.24982, 296.28497,
            297.2611, 297.2795, 297.30798, 298.265, 298.2893, 298.29938, 299.34207, 299.3439,
            299.36365, 299.38684, 300.32654, 300.34442, 300.3585, 301.18237, 301.34628, 301.36996,
            302.31207, 302.32614, 302.34192, 302.34283, 303.30814, 303.32306, 303.34802, 304.32675,
            304.34723, 305.1874, 305.34937, 305.36835, 306.18634, 306.19897, 306.3686, 307.18417,
            307.20242, 307.36926, 308.24564, 308.24655, 308.31924, 311.18417, 311.20007, 312.22336,
            316.2408, 319.22055, 319.23538, 319.24332, 319.28162, 320.26508, 321.2286, 321.24414,
            321.2779, 322.23575, 322.23996, 322.26297, 322.28024, 323.2423, 324.18356, 324.19934,
            324.2202, 325.22537, 325.2263, 325.2394, 325.26254, 326.2797, 326.29575, 327.28586,
            327.30432, 327.32407, 328.28586, 328.28775, 329.30374, 329.32184, 330.30502, 330.32245,
            330.33923, 331.17816, 331.19772, 331.1996, 331.2173, 332.25943, 334.165, 337.1944,
            340.17984, 343.1584, 346.17612, 349.24045, 349.24136, 349.26685, 349.2785, 350.1628,
            350.18292, 350.198, 351.15976, 351.1728, 351.20047, 352.21954, 352.2416, 352.24252,
            352.26224, 353.1589, 353.17657, 353.1988, 353.24042, 353.28314, 353.2975, 354.198,
            354.2399, 355.2397, 355.25427, 356.20447, 356.22006, 356.2378, 356.24335, 356.25958,
            357.18066, 357.2364, 357.25885, 357.28036, 358.1931, 358.2197, 358.24158, 359.21564,
            359.26642, 359.28583, 360.22556, 360.24487, 360.28036, 361.25977, 361.28595, 362.15714,
            364.1424, 364.15274, 364.17444, 365.1356, 365.15588, 365.17993, 365.18085, 367.15826,
            367.17706, 367.20322, 368.13647, 368.15656, 368.1584, 368.1825, 369.1339, 369.15686,
            369.179, 370.1429, 370.15485, 370.25916, 371.1376, 371.15826, 371.15918, 371.17935,
            372.1418, 372.15417, 372.18033, 373.1278, 373.1587, 373.18103, 374.13275, 374.15768,
            374.15857, 375.17545, 375.1987, 376.16278, 376.174, 376.19482, 377.13638, 377.1373,
            377.16132, 377.1791, 378.136, 378.21683, 378.24744, 379.21725, 379.2388, 381.1212,
            381.13916, 381.1415, 381.15912, 382.14224, 382.15582, 383.1172, 383.13477, 383.15802,
            384.1447, 384.15216, 384.17374, 385.19153, 385.21136, 386.20087, 386.21588, 387.11957,
            387.12048, 387.13766, 387.15375, 389.2162, 390.20016, 390.21588, 390.2168, 423.07803,
            476.53094,
        ];
        let y = [
            16.585, 16.587, 16.592, 16.617, 16.6, 16.602, 16.611, 16.566, 16.577, 16.59, 16.592,
            16.591, 16.582, 16.577, 16.586, 16.576, 16.576, 16.56, 16.599, 16.581, 16.596, 16.592,
            16.593, 16.651, 16.587, 16.604, 16.579, 16.579, 16.591, 16.564, 16.602, 16.578, 16.588,
            16.598, 16.573, 16.579, 16.572, 16.59, 16.598, 16.614, 16.596, 16.577, 16.591, 16.577,
            16.598, 16.574, 16.642, 16.597, 16.614, 16.597, 16.606, 16.583, 16.599, 16.592, 16.602,
            16.6, 16.558, 16.569, 16.569, 16.598, 16.617, 16.588, 16.611, 16.602, 16.625, 16.613,
            16.582, 16.594, 16.612, 16.602, 16.625, 16.596, 16.606, 16.6, 16.611, 16.603, 16.599,
            16.583, 16.582, 16.58, 16.599, 16.595, 16.613, 16.586, 16.627, 16.587, 16.594, 16.568,
            16.601, 16.601, 16.604, 16.589, 16.597, 16.579, 16.581, 16.591, 16.586, 16.586, 16.575,
            16.609, 16.593, 16.59, 16.575, 16.572, 16.597, 16.569, 16.577, 16.595, 16.591, 16.575,
            16.589, 16.589, 16.582, 16.579, 16.601, 16.589, 16.57, 16.584, 16.587, 16.593, 16.586,
            16.593, 16.573, 16.594, 16.593, 16.595, 16.594, 16.59, 16.579, 16.575, 16.571, 16.573,
            16.595, 16.591, 16.561, 16.585, 16.606, 16.57, 16.588, 16.592, 16.579, 16.597, 16.597,
            16.565, 16.6, 16.569, 16.57, 16.592, 16.584, 16.585, 16.588, 16.595, 16.569, 16.59,
            16.598, 16.592, 16.608, 16.573, 16.557, 16.575, 16.569, 16.569, 16.579, 16.599, 16.605,
            16.589, 16.58, 16.576, 16.57, 16.559, 16.565, 16.606, 16.58, 16.578, 16.573, 16.591,
            16.612, 16.575, 16.609, 16.557, 16.592, 16.589, 16.598, 16.61, 16.576, 16.567, 16.588,
            16.592, 16.614, 16.595, 16.601, 16.58, 16.581, 16.598, 16.616, 16.579, 16.57, 16.573,
            16.571, 16.588, 16.577, 16.598, 16.602, 16.569, 16.591, 16.584, 16.575, 16.587, 16.532,
            16.552, 16.598, 16.566, 16.589, 16.582, 16.563, 16.603, 16.638, 16.629, 16.591, 16.578,
            16.595, 16.59, 16.59, 16.582, 16.553, 16.576, 16.578, 16.563, 16.59, 16.604, 16.548,
            16.575, 16.583, 16.576, 16.574, 16.595, 16.563, 16.554, 16.558, 16.567, 16.585, 16.61,
            16.581, 16.596, 16.555, 16.564, 16.559, 16.569, 16.596, 16.585, 16.564, 16.541, 16.561,
            16.536, 16.589, 16.579, 16.549, 16.585, 16.562, 16.519, 16.564, 16.566, 16.555, 16.564,
            16.607, 16.565, 16.57, 16.591, 16.562, 16.599, 16.585, 16.557, 16.616, 16.605, 16.596,
            16.602, 16.586, 16.575, 16.578, 16.621, 16.591, 16.604, 16.609, 16.599, 16.612, 16.578,
            16.62, 16.574, 16.596, 16.588, 16.604, 16.588, 16.586, 16.58, 16.594, 16.587, 16.587,
            16.585, 16.577, 16.573, 16.584, 16.588, 16.572, 16.589, 16.563, 16.576, 16.594, 16.61,
            16.579, 16.59, 16.589, 16.562, 16.591, 16.556, 16.584, 16.586, 16.586, 16.578, 16.596,
            16.597, 16.573, 16.598, 16.593, 16.546, 16.583, 16.577, 16.573, 16.591, 16.607, 16.572,
            16.55, 16.573, 16.58, 16.551, 16.592, 16.572, 16.557, 16.554, 16.622, 16.587, 16.614,
            16.582, 16.636, 16.581, 16.597, 16.595, 16.573, 16.595, 16.612, 16.578, 16.554, 16.586,
            16.586, 16.585, 16.583, 16.662, 16.613, 16.607, 16.592, 16.603, 16.608,
        ];
        let mut ts: TimeSeries<f32> = TimeSeries::new_without_weight(&x, &y);
        let actual = eval.eval(&mut ts).unwrap();
        assert!(actual.iter().all(|x| x.is_finite()));
    }
}
