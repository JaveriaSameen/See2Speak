See2Speak: A Computer Vision-Based Vocabulary Learning Game 
Computer Vision (DSP487/M-30291) Project Proposal
Group Members:
Areej Mazhar
20231-34638
Javeria Sameen
20231-34722 
Rafay Khan
20231-34594 


Abstract
See2Speak is an interactive English vocabulary game that combines computer vision, speech recognition, and facial landmark analysis to help children practise everyday English words. Using a laptop webcam, the pretrained YOLO-World v2 Small model detects objects from a predefined list, including books, chairs, cell phones, apples, pens, and pencils, and displays their names and bounding boxes on the screen. The child then says the detected object's name, which is transcribed using the pretrained OpenAI Whisper base.en model. The recognized text is normalized and compared with the target word using exact matching and edit distance to assess the response.
The system also uses MediaPipe Face Landmarker to measure mouth opening and width during speech as a basic check for visible mouth movement. This check does not perform lip reading and cannot reliably distinguish genuine speech from all prerecorded audio or lip-sync attempts. Points and progress tracking encourage continued practice. All core models use pretrained weights without custom model training or fine-tuning and are intended to run locally on a laptop. The system will be evaluated using collected webcam images and audio-video recordings under different lighting and background conditions to measure object detection performance, speech matching accuracy, mouth-movement check reliability, processing speed, and usability. The project focuses on integrating and evaluating these components while recognizing limitations in detecting small or thin objects and assessing spoken responses.
Introduction
Problem Statement
Traditional vocabulary learning often relies on flashcards and memorization, offering limited opportunities to connect English words with real-world objects and practise speaking. Audio-only tools can transcribe spoken responses but may not check for visible mouth movement during an attempt. There is a need for an interactive learning tool that combines real-world object recognition, spoken-word practice, and a basic mouth-movement check while acknowledging that these methods cannot independently verify the source of the audio. 
Proposed Solution
See2Speak is an interactive English vocabulary game that uses a laptop webcam to detect selected everyday objects, such as books, chairs, cell phones, apples, pens, and pencils. It displays the detected object's English name and plays its pronunciation. The user then says the word, and the system transcribes the response and compares it with the target word to provide feedback. Facial landmark tracking measures mouth opening and width during the attempt, providing an additional check for visible mouth movement. However, this check cannot reliably distinguish genuine speech from every prerecorded-audio or lip-sync attempt. Points and progress tracking encourage continued practice. The system integrates pretrained models without custom training and will be evaluated for object detection, speech matching, mouth-movement checking, processing speed, and usability under everyday conditions.
Literature Review
Literature Review Table
The following table summarizes the relevant research, its methodology, findings, limitations, and relevance to See2Speak. 
No.
Research Paper
What the Paper Introduces
Limitations
Relevance to See2Speak
1
Fei et al. (2022), CollectiAR: Computer Vision-Based Word Hunt for Children with Dyslexia [1] 
Introduces a computer vision word-hunt game that replaces flashcards by letting children scan real-world objects to hear and spell English words. 
Uses a closed-set detector (~70.6% accuracy) and focuses on listening and spelling rather than spoken word practice. 
Supports replacing flashcards with real-world object detection while showing the need to add active speaking practice. 
2
Bashori et al. (2022), 'Look, I can speak correctly': learning vocabulary and pronunciation through websites equipped with automatic speech recognition technology [2] 
Shows that automatic speech recognition (ASR) websites improve vocabulary retention and encourage active speaking practice. 
Relies on audio-only input without real-world object detection or any camera check for visible mouth movement. 
Supports using speech recognition for spoken practice while showing the limitation of audio-only tools that cannot verify visible mouth movement. 
3
Guy et al. (2021), Learning Visual Voice Activity Detection with an Automatically Annotated Dataset [3] 
Uses 2D facial landmarks to detect visible mouth movement during speech (Visual Voice Activity Detection). 
Detects mouth movement rather than specific words (not lip reading) and cannot verify if concurrent audio is live or prerecorded. 
Supports using facial landmarks as a basic mouth-movement check while recognizing it cannot independently verify the audio source. 


Research gap identified
Existing educational tools show that computer vision connects words to real-world objects [1] and speech recognition supports active speaking practice [2], while visual voice activity detection tracks mouth movement [3]. However, vision-based vocabulary games often lack spoken response checks [1], and audio-only tools cannot check for visible mouth movement [2]. See2Speak bridges this gap by combining pretrained open-vocabulary object detection [4], speech recognition [5], and facial landmark tracking [6] in a single vocabulary game and evaluating how well they work together on a standard laptop. 

Methodology


Datasets and Models
Object detection: Use Ultralytics YOLO-World v2 Small (yolov8s-worldv2.pt), a pretrained open-vocabulary object detector that supports custom text labels without fine-tuning. Cheng et al. (2024) [4] introduced YOLO-World to address the limitations of detectors restricted to fixed object categories, while aiming for efficient detection. This supports our choice because See2Speak needs to recognize selected everyday objects, including books, chairs, cell phones, apples, pens, and pencils, without training a custom model. The application will display detected object names and bounding boxes on the webcam feed. Detection accuracy and speed will depend on object visibility, size, distance, and laptop performance.
Speech recognition: Use OpenAI Whisper base.en, a pretrained English speech recognition model running locally on the laptop CPU. Radford et al. (2023) [5] demonstrated that models trained on large-scale audio and transcripts can generalize across different speech recognition tasks without dataset-specific fine-tuning. This supports using a pretrained model to transcribe spoken vocabulary words without training a new speech recognition model. If inference is too slow, test tiny.en. Normalize the transcript and compare it with the target word using exact matching and edit distance. Transcription matching will assess whether the recognized word matches the target, not whether its pronunciation is accurate.
Mouth landmark analysis: Use the free MediaPipe Face Landmarker model to track inner-lip movement at typical laptop webcam distance. Pan et al. (2022) [6] proposed a real-time approach to face detection and facial landmark localization, supporting the use of facial landmarks for analyzing facial geometry. In See2Speak, this motivates measuring mouth opening and width relative to facial landmarks, using a neutral-mouth baseline and simple thresholds to check for visible movement during speech. This is a supplementary check, not lip reading or reliable verification of the audio source.
Evaluation dataset: Collect a small set of webcam images and audio-video clips under varied lighting, backgrounds, and ordinary desk conditions. Include correct spoken responses, mispronunciation, silence, and prerecorded audio. Keep any data used to tune thresholds separate from the final test recordings where feasible. These recordings will evaluate the integrated components rather than train or fine-tune the pretrained models.
Expected Results
See2Speak aims to provide an interactive English vocabulary learning experience by combining object detection, speech recognition, and mouth landmark analysis.
Object Detection: Detect vocabulary objects through the webcam with at least 80% correct labels for clearly visible objects in normal lighting (at least 70% in dim or cluttered scenes), reporting accuracy per object and frame rate.
Vocabulary Learning: Link real-world objects to their English names and pronunciations, showing the word within 3 seconds of locking onto an object. 
Speech Practice: Accept at least 80% of correct pronunciations in a quiet room, and report how often deliberate mispronunciations are still accepted. 
Mouth Movement Check: Accept at least 85% of genuine attempts and reject at least 85% of silent or played-back attempts where the user’s mouth stays still. Simple lip-closure rules (for words starting with b, p or m) should agree with manual labels at least 80% of the time. 
Engagement and Progress: Points, feedback, and recorded scores to monitor learning progress.
System Evaluation: Detection accuracy, response time, speech matching, mouth-check accuracy, and usability through a small user test.
Conclusion
See2Speak combines object detection, speech recognition, and facial landmark analysis to make English vocabulary practice more interactive. The mouth-movement check provides an additional signal during speaking attempts but cannot independently verify the source of audio or assess pronunciation accurately. The project will evaluate detection performance, word matching, mouth-check effectiveness, response time, and usability. Future work could explore word-level lip reading and more advanced pronunciation assessment.
References
[1] D. Fei, Z. Gao, L. Yuan, and Z. A. Wen, “CollectiAR: Computer Vision-Based Word Hunt for Children with Dyslexia,” in Extended Abstracts of the 2022 Annual Symposium on Computer-Human Interaction in Play (CHI PLAY '22), 2022, pp. 158–164. doi: 10.1145/3505270.3558318.
[2] M. Bashori, R. van Hout, H. Strik, and C. Cucchiarini, “‘Look, I can speak correctly’: learning vocabulary and pronunciation through websites equipped with automatic speech recognition technology,” Comput. Assist. Lang. Learn., vol. 37, no. 5–6, pp. 1335–1363, 2022. doi: 10.1080/09588221.2022.2080230.
[3] S. Guy, S. Lathuilière, P. Mesejo, and R. Horaud, “Learning Visual Voice Activity Detection with an Automatically Annotated Dataset,” in Proc. 25th Int. Conf. Pattern Recognit. (ICPR), 2021, pp. 4851–4856. doi: 10.1109/ICPR48806.2021.9412884.
[4] T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, and Y. Shan, “YOLO-World: Real-Time Open-Vocabulary Object Detection,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 16901–16911. doi: 10.1109/CVPR52733.2024.01599.
[5] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust Speech Recognition via Large-Scale Weak Supervision,” in Proc. 40th Int. Conf. Mach. Learn., vol. 202, 2023, pp. 28492–28518. doi: 10.48550/arXiv.2212.04356.
[6] Z. Pan, Y. Wang, and S. Zhang, “Joint Face Detection and Facial Landmark Localization Using Graph Match and Pseudo Label,” Signal Process.: Image Commun., vol. 102, Art. no. 116587, 2022. doi: 10.1016/j.image.2021.116587

