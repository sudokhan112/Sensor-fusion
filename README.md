# Sensor-fusion

Combining inputs from different sensors at decision level to achieve correct object classification, we need a robust decision level sensor fusion algorithm. Sensor fusion can be represented at three different levels. Signal level can be explained if raw pixels from multiple cameras are combined. Feature level sensor fusion can be gained by hard-coded features (area or moment of certain object) are extracted from images. Final decision is made using the output of another algorithm. At decision level, class ID identification is already made by using a supervised/unsupervised algorithm. Then decisions from multiple sensors are fused using information fusion algorithm. Decision level fusion approach can increase the final classification accuracy by taking advantage of the best classification result from one of the sensors from the sensor array. At decision level, a crucial issue in multi-source information fusion is, how to represent and determine the imprecise, fuzzy, ambiguous, inconsistent, and even incomplete information. As a tool to manipulate an uncertain environment, Dempster–Shafer evidence theory is an established system for uncertainty management.

<p>
    <img src="https://github.com/sudokhan112/Sensor-fusion/blob/main/2.png" width=700 height=400 alt>
    <em>Levels of sensor fusion</em>
</p>

In a perfect world, the moment a CNN classifier sees an object, it should classify that object with 100% classification accuracy. But in real world, the classification accuracy varies a lot due to noise, lighting, vibration, occlusion etc. A steady output is needed from object classification system to spray system to spray herbicides accurately. A multi-sensor fusion architecture with capability to fuse evidence in time and space domain is proposed to create a steady classification output. In time domain, a certain number of time-steps is chosen to fuse the evidences for each sensor. Because evidences are weighted by the fusion algorithm, if at a certain time step, a sensor output disagrees with other time steps (wrong classification), it will be given less weight. If higher number of time-steps is considered for time domain fusion, the output will be more robust against wrong classification (smoother curve) but the response time (rate of change of classification output) will be slower. An artistic representation of how multiple cameras can be placed on an AgBot is shown here.

<p>
    <img src="https://github.com/sudokhan112/Sensor-fusion/blob/main/fig_22.png" width=700 height=400 alt>
    <em>Placement of multiple cameras on an AgBot</em>
</p>

With increasing number of sensors, the possibility of finding a faulty sensor also increases. Any multi-sensor fusion algorithm should be able to find the faulty sensor in the system and compensate for that in fusion architecture. The proposed algorithm is able to find the faulty sensor in space domain fusion step because faulty sensor disagrees with the evidences from other sensors. The algorithm compensates for that by applying less weight to the faulty sensor evidences so that it has less effect on final classification output. Following figures show the problem and output of proposed algorithm. 

<p>
    <img src="https://github.com/sudokhan112/Sensor-fusion/blob/main/1.png" width=700 height=400 alt>
    <em>Reduced unstable classification with time domain sensor fusion. Reduce the effect of faulty sensor evidence with space domain sensor fusion.</em>
</p>

<p>
    <img src="https://github.com/sudokhan112/Sensor-fusion/blob/main/fig15new.png" width=700 height=400 alt>
    <em>Ground truth value for Ragweed and Pigweed at the top. (a) Classification accuracy from left, center and right camera for Pigweed. (b) Classification accuracy from left, center and right camera for Rigweed. (c) Reduced unstable output (smooth curve) with time domain fusion. Each line shows the classification % of a specific weed for a specific camera. (d) Eliminated the effect of faulty sensor (right camera) evidence on final classification output with space domain fusion.</em>
</p>


 The goal is to observe how the fusion algorithm handles the faulty evidence from right camera. Right camera is showing higher accuracy of Pigweed for time steps 20 - 80 (where it should be high accuracy for Ragweed from ground truth) and for time steps 100 - 150 showing higher accuracy for Ragweed (where it should be Pigweed). After time step fusion, unstable outputs are reduced (smoother curve) but right camera still showing wrong output. After space fusion, the faulty evidence from right camera is compensated with lower weights. Final output shows steady and accurate weed classification depicting ground truth values. 
 
**[Link to PhD thesis defense](https://www.youtube.com/watch?v=IX1noD8NZBY)**


**[Link to Thesis pdf](https://github.com/sudokhan112/Sensor-fusion/blob/main/Purdue_University_Thesis_PhD_Nazmuzzaman.pdf)**

### To Learn more about Space and Time domain fusion from our published work:

**[Space domain fusion](https://www.mdpi.com/1424-8220/19/21/4810)**

**[Time domain fusion](https://www.mdpi.com/1424-8220/19/23/5187)**
