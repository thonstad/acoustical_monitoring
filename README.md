[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/thonstad/acoustical_monitoring)

# Acoustical monitoring framework for damage detection in concrete structures during earthquakes.

## Vision
Currently, the assessment of the integrity and safety of structures subjected to earthquakes relies on a combination of visual inspection and engineering judgement. This process is time consuming and can miss critical damage that is not visible. For example, the damage to reinforcing steel in concrete structures can be concealed by concrete cover. In steel buildings, the structural components are often hidden by architectural features and fire proofing.
New approaches are needed to rapidly and reliably inform decisions about closures and restrictions in service for bridges and buildings. In bridges, the damage needs to be identified quickly, because they are often critical lifelines in the post-hazard response and recovery. Building damage needs to be identified quickly too, because they are needed to house people and support the recovery.
One promising strategy for the accurate and timely assessment of earthquake damage is to instrument structures with numerous, low-cost, acoustical sensors. Such acoustical monitoring of structures has not been developed for earthquake engineering applications. The objective of the proposed research is develop the intensive data processing and management strategies needed to support the use of acoustical sensors to “listen” for earthquake damage.

## Objectives:
We want to generate a proof of concept methodology for the use of acoustic monitoring to detect rebar fractures in concrete structures during earthquakes. The project will focus on data from a single experiment, but the developed processing and analysis methods should be general enough to apply to other experiments.
In the duration of the incubator, we want to be able to :
* Establish a ground truth and determine ways of evaluating different methodologies against this ground truth. When there is no ground truth available, develop a framework to facilitate qualitative evaluation of the performance of the methods. For example, this framework could present the prediction results superimposed on the raw audio data for manual comparison.
* Organize data into an appropriate structure for analysis. Automate the pre-processing tasks to convert raw video files to audio streams. This process would include extracting and synchronizing the audio signals.
* Study methods to distinguish mechanical noise from damage to the structure. Possible approaches would include; analysis of the correlation structure of the time series, testing for differences between ‘no fracture’ sequences vs ‘fracture’ sequences, feature detection of specific acoustic signature (interferences from other equipment).
* Provide a first analysis whether solely removing the noise can allow detection of fractures using simple techniques such as thresholding. 
* If this is not sufficient, over the course of the incubator we will explore other, more complex methods to detect fracture. This may include using the structure of the time series, or the influence of external parameters related to the experiment set-up. For example, we would like to see whether incorporating prior (fatigue) models can improve detection relative to using observed data alone.
* Determine whether incorporating information from several cameras can improve fracture detection, and whether these multiple data streams can be used to locate the most probable locations of the fractures within the structure.    
* Collect suitable projects from the NEEShub Project Warehouse (nees.org) that could be used to further validate and refine detection methods. Publish these datasets so that other researchers can access and use the processed data.
* Define a roadmap for further developments and refinements that should be envisaged to improve the detection methodology. The robust acoustical monitoring strategy that we want to develop will not be completed in a single quarter, and we want to come out with a clear roadmap for further work on this complex problem.

## Success criteria:
* We are able to automate all pre-processing tasks.
* We are able to sync the extracted audio data so that there are discernable, consistent time-of-arrival differences between instruments.
* We can detect the 1 rebar fracture heard in Motion 17 and have 0 fractures in prior motions. 
* We can determine the 72 most probable fracture events during the experimental schedule.
* We have determined whether using multiple cameras rather than a single camera is beneficial in detecting fractures, and whether it can be used to locate instances of rebar fracture.
* We have shown the robustness of the chosen methodology to perturbations in data
* We have a set of suitable experiments from the NEEShub Project Warehouse to further refine and develop fracture detection algorithms.
* We have a roadmap for generalization of the methods and tools developed
* We have an interactive data visualization demonstration.

## Deliverable Schedule:
* **Jan 14th:** Data for experiments summarized, and initial set-up of code, data, documentation  repositories planned
* **Jan 21st:** Automated preprocessing steps,determine asynchronicity of audio files, review previous acoustical fracture detection algorithm, created ground truth dataset
* **Feb  11:** Perform Noise Analysis, and attempt noise removal for fracture detection, signal separation analysis, and deliver a report (ipython notebook/blog) on the discoveries.
* **Feb 18:** Preliminary comparison of methods based on single camera vs multiple cameras          
* **Feb 25th:** Summary of potential external datasets that can be used for acoustic monitoring. Robustness analysis of the methodology on the PreT dataset.
* **March 1:** Document with roadmap for for further developments and refinements.
* **March 8:** A document describing the assessment of the developed methodologies and their performance is available.
* **March 13:** Final Code Implementation tested.
* **March 16:** Final Poster/Presentation/Documentation.
