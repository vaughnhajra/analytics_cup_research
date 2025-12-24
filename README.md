# SkillCorner X PySport Analytics Cup

Vaughn Hajra, 12/24/2025

## Understanding the Spatial Drivers of On-Field Reaction Time
#### Introduction

Despite game context being central to on-field reactions, traditional reaction time is only measured in a controlled environment. To address this gap, I introduce a novel framework to measure on-field **defensive response latency** and **defensive response frequency**. Derived from raw tracking data, I extract player-level reaction events, identify the spatial drivers of response timing, and develop probabilistic player response profiles.


#### Methods

I first smooth raw tracking data with a Savitzky-Golay filter. I detect ball-initiated “stimulus events” using sharp changes in acceleration (≥ 15 m/s<sup>2</sup>) and subsequent deceleration (≤ -10 m/s<sup>2</sup>). A player’s response is identified as their first deviation in instantaneous acceleration (≥ 3.5 m/s<sup>2</sup>) following the stimulus.

- **Defensive Response Latency:** The time between the ball’s acceleration event and the player’s qualifying acceleration change.

- **Defensive Response Frequency:** The proportion of all qualifying stimulus events to which a defender responds.
<br>
<br>

To ensure validity, I restrict potential responses to defenders within 50 meters and 1.25 seconds of the ball’s location at event onset. A Shapiro-Wilk test confirms non-normality in latency (W = 0.972, p < 0.001) and a Kruskal-Wallis shows evidence of heterogeneity across players (H = 165.571, p = 0.073). 

To identify the drivers of defensive response latency, I fit complementary models. I first use OLS regression as an interpretable baseline. I then fit a random forest model to capture nonlinear interactions and higher-order effects. Finally, I apply a Gaussian Mixture Model to generate soft player clusters.


#### Results

![Figure 1](figures/Figure%201.png)

The OLS regression identifies several spatial and contextual factors as significant predictors of defensive reaction time (overall F-test p < 0.001). Responses are faster when ball movement is rapid, suggesting that stimulus salience drives quicker reactions. Response times are slower when passes end closer to the goal, where poor commitments can be costly. Players also respond more quickly when individual responsibility is heightened, captured by proximity to a pass’ intended receiver and last-defender status. While the linear model explains a modest share of variance (R^2 = 0.065), the random forest improves explanatory power (R^2 = 0.15), indicating that defensive behavior is shaped by complex, non-additive spatial dynamics.

![Figure 2](figures/Figure%202.png)

The Gaussian Mixture Model reveals defensive response profiles. One group reacts ![quickly and selectively](https://img.shields.io/badge/-quickly%20and%20selectively-%232e7d32?style=flat-square)
, another reacts ![slower and selectively](https://img.shields.io/badge/-slower%20and%20selectively-%230d47a1?style=flat-square)
, and the final reacts ![moderately fast and most frequently](https://img.shields.io/badge/-moderately%20fast%20and%20most%20frequently-%23c62828?style=flat-square). This mixture model approach allows each player to be captured by a unique combination of these clusters, representing individual defensive playstyles.

#### Conclusion

This project shows that in-game defensive responses can be quantified directly from tracking data, revealing how spatial context shapes on-field decision-making. By linking interpretable regression results with machine learning, this framework moves beyond static reaction tests toward game-realistic measures of defensive cognition. 

Open-sourcing this work supports the integration of richer inputs (formations, tactics, and player roles) and enables broader identification of defensive profiles as new cognitive metrics are created. This will allow future research to connect individual behavior to team success.
