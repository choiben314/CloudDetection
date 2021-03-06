#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/ccalib/omnidir.hpp"

#define _USE_MATH_DEFINES
#include "math.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

//const string VIDEO_PATH = "D:/NIFA/lamberton_08_10_2020/DJI_0001.MOV";
//const string VIDEO_PATH = "E:/NIFA/footage/elgin_07_27_2020/DJI_0004.MOV";
const string VIDEO_PATH = "E:/NIFA/footage/grand_meadow_07_08_2020/DJI_0001.MOV";
// const string VIDEO_PATH = "E:/NIFA/becker_07_24_2020/DJI_0004.MOV";
//const string VIDEO_PATH = "D:/NIFA/elgin_07_27_2020/DJI_0001.MOV";
//const string VIDEO_PATH = "D:/NIFA/lamberton_08_10_2020/DJI_0001.MOV";
//const string ROT_REF_PATH = "D:/NIFA/rotation_test_images/reference.jpg";
//const string ROT_ROTATED_PATH = "D:/NIFA/rotation_test_images/rotated.jpg";

string OMNI_MODEL_PATH = "E:/NIFA/calibration/camera_models/live_fisheye_model.xml";
const string GCP_LOCATION = "E:/NIFA/footage/grand_meadow_07_08_2020/";
const string GCP_PATH = GCP_LOCATION + "gcp.xml";
const string GCP_PIXEL_COORDS_PATH = GCP_LOCATION + "pixel_coords.xml";


//const float RESIZE_FACTOR = 1.5;

// Get image coordinates of fiducial markers defined in GCP_PATH
static void markFiducials(Mat& world_coords, vector<Point2f>& pixel_coords) {
	FileStorage gcp_read(GCP_PATH, FileStorage::READ);
	gcp_read["gcp"] >> world_coords;

	FileStorage fs(GCP_PIXEL_COORDS_PATH, FileStorage::READ);
	if (fs.isOpened()) {
		fs["pixel_coords"] >> pixel_coords;
	}
}

// Get camera pose via PNP solver
static void estimateCameraPose(const Mat& world_coords, const vector<Point2f>& pixel_coords, const Mat& K, const Mat& D, Mat& rvec, Mat& tvec, vector<Point2d>& new_pixel_coords) {
	solvePnPRansac(world_coords, pixel_coords, K, D, rvec, tvec);
	projectPoints(world_coords, rvec, tvec, K, D, new_pixel_coords);
}

// Helper function: Get evenly spaced coordinates on a line between two points
void generateLineCoordinatesHelper(const Point3f& start, const Point3f& end, int numCoords, vector<Point3f>& line_coords) {
	if (start.x == end.x) {
		float total_distance = end.y - start.y;
		for (float i = 0; i < numCoords; i++) {
			float frac = i / (numCoords - 1);
			line_coords.push_back(Point3f(start.x, start.y + frac * total_distance, 0));
		}
	}
	else if (start.y == end.y) {
		float total_distance = end.x - start.x;
		for (float i = 0; i < numCoords; i++) {
			float frac = i / (numCoords - 1);
			line_coords.push_back(Point3f(start.x + frac * total_distance, start.y, 0));
		}
	}
	else {
		cout << "One set of corresponding coordinates must be equal." << endl;
	}
}


// Get coordinates for sampling on East-North plane
void generateENZPlaneCoordinates(const Point3f& min_corner, const Point3f& max_corner, const Size& size, Mat& K, Mat& D, Mat& rvec, Mat& tvec, vector<Point2f>& new_enz_coords) {
	Point3f top_left = Point3f(min_corner.x, max_corner.y, min_corner.z);
	vector<Point3f> left_edge;
	generateLineCoordinatesHelper(min_corner, top_left, size.height, left_edge);
	vector<Point3f> enz_coords;
	for (int i = 0; i < left_edge.size(); i++) {
		vector<Point3f> row;
		generateLineCoordinatesHelper(left_edge[i], Point3f(max_corner.x, left_edge[i].y, min_corner.z), size.width, row);

		for (int j = 0; j < row.size(); j++) {
			enz_coords.push_back(row[j]);
		}
	}
	projectPoints(enz_coords, rvec, tvec, K, D, new_enz_coords);
}

// Helper function: Bilinear interpolation for image pixels
cv::Vec3b getColorSubpixHelper(const cv::Mat& img, cv::Point2f pt)
{
	cv::Mat patch;
	cv::getRectSubPix(img, cv::Size(1, 1), pt, patch);
	return patch.at<cv::Vec3b>(0, 0);
}

// Get and display ENZ sampling region coordinates
void getSamplingRegion(const Mat& frame, Point3f& min_corner, Point3f& max_corner, Size dim, Mat& K, Mat& D, Mat& rvec, Mat& tvec, bool show, vector<Point2f>& enz_coords) {
	generateENZPlaneCoordinates(min_corner, max_corner, dim, K, D, rvec, tvec, enz_coords);
	if (show) {
		for (int i = 0; i < enz_coords.size(); i++) {
			circle(frame, enz_coords[i], 3, Scalar(255, 0, 0), -1);
		}
		//imshow("Sampling region", frame);
	}
}

// Get image resampled to ENZ
void getRegisteredImage(const Mat& frame, vector<Point2f>& enz_coords, bool show, Mat& new_frame) {
	vector<Vec3b> registered;
	for (int i = 0; i < enz_coords.size(); i++) {
		registered.push_back(getColorSubpixHelper(frame, enz_coords[i]));
	}
	new_frame = Mat(registered).reshape(3, 1400);
	rotate(new_frame, new_frame, ROTATE_90_COUNTERCLOCKWISE);
	if (show) {
		imshow("Registered and resampled", new_frame);
	}
}

void getSurfRotation(const Mat& ref_color, const Mat& rot_color, Mat& H) {
	Mat img_ref, img_rot; 
	cvtColor(ref_color, img_ref, COLOR_BGR2GRAY);
	cvtColor(rot_color, img_rot, COLOR_BGR2GRAY);

	int minHessian = 400;
	Ptr<SURF> detector = SURF::create(minHessian);

	vector<KeyPoint> keypoints_ref, keypoints_rot;
	Mat descriptors_ref, descriptors_rot;
	detector->detectAndCompute(img_ref, noArray(), keypoints_ref, descriptors_ref);
	detector->detectAndCompute(img_rot, noArray(), keypoints_rot, descriptors_rot);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	vector< vector<DMatch> > knn_matches;
	matcher->knnMatch(descriptors_ref, descriptors_rot, knn_matches, 2);

	const float ratio_thresh = 0.75f;
	vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}

	//Mat img_matches;
	//drawMatches(img_ref, keypoints_ref, img_rot, keypoints_rot, good_matches, img_matches, Scalar::all(-1),
	//	Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	vector<Point2f> ref;
	vector<Point2f> rot;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		ref.push_back(keypoints_ref[good_matches[i].queryIdx].pt);
		rot.push_back(keypoints_rot[good_matches[i].trainIdx].pt);
	}
	//-- Get the corners from the image_1 ( the ref to be "detected" )
	//vector<Point2f> obj_corners(4);
	//obj_corners[0] = Point2f(0, 0);
	//obj_corners[1] = Point2f((float)img_ref.cols, 0);
	//obj_corners[2] = Point2f((float)img_ref.cols, (float)img_ref.rows);
	//obj_corners[3] = Point2f(0, (float)img_ref.rows);
	//vector<Point2f> rot_corners(4);
	//perspectiveTransform(obj_corners, rot_corners, H);

	//-- Draw lines between the corners (the mapped ref in the rot - image_2 )
	//line(img_matches, rot_corners[0] + Point2f((float)img_ref.cols, 0),
	//	rot_corners[1] + Point2f((float)img_ref.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_matches, rot_corners[1] + Point2f((float)img_ref.cols, 0),
	//	rot_corners[2] + Point2f((float)img_ref.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_matches, rot_corners[2] + Point2f((float)img_ref.cols, 0),
	//	rot_corners[3] + Point2f((float)img_ref.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_matches, rot_corners[3] + Point2f((float)img_ref.cols, 0),
	//	rot_corners[0] + Point2f((float)img_ref.cols, 0), Scalar(0, 255, 0), 4);

	//H = findHomography(rot, ref, RANSAC);
	H = estimateAffinePartial2D(rot, ref);
	//Mat new_rot;
	//warpPerspective(img_rot, new_rot, H, img_ref.size());
	//imshow("Rotated Image", new_rot);

	//imshow("Good Matches & Object detection", img_matches);

	//while (1) {
	//	namedWindow("Good Matches & Object detection", WINDOW_NORMAL);
	//	resizeWindow("Good Matches & Object detection", img_matches.cols / 1.5, img_matches.rows / 1.5);
	//	imshow("Good Matches & Object detection", img_matches);
	//	waitKey(1);
	//}
}

void getBinaryCloudMask(const Mat& img, Mat& bright, Mat& binary) {
	Mat hsv, v, diff, very_dark;
	vector<Mat> hsv_channels;

	cvtColor(img, hsv, COLOR_BGR2HSV);
	split(hsv, hsv_channels);
	v = hsv_channels[2];

	// Running maximum brightness for each pixel
	max(bright, v, bright);

	//// Anything lower than threshold is zero, everything else is one
	//threshold(v, very_dark, 30, 1, THRESH_BINARY);

	//// Make super dark pixels ineligible to be clouds
	//multiply(bright, very_dark, bright);

	// Difference between brightest observed and current
	subtract(bright, v, diff);

	// Anything with too small of a difference is floored to zero (Avoids segmentation when there is no segmentation to be done). 
	threshold(diff, diff, 20, 255, THRESH_TOZERO);

	// Dynamic thresholding to binary by minimizing within-class variance 
	threshold(diff, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
}
const Size FINAL_SIZE = Size(1280, 720);

// Downscale images if using original 4K footage
void downscale(Mat& old_frame, Mat& new_frame) {
	pyrDown(old_frame, new_frame);
	pyrDown(new_frame, new_frame);
	resize(new_frame, new_frame, FINAL_SIZE);
}

int main(int argc, char** argv) {
	VideoCapture cap;
	cap.open(VIDEO_PATH);
	cout << "SDLKJFLKDSJF" << endl;
	if (!cap.isOpened())
		return -1;
	cout << "WHLJELKSJDF";
	int num_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
    cout << num_frames << " frames." << endl;

	//Mat frame1;
	//cap >> frame1;
	//imwrite("D:/NIFA/frame1.JPG", frame1);
	//return 0;

	//cap.set(CAP_PROP_POS_FRAMES, 6000);
	int	count = 1;

	Mat frame, img_ref, img_rot, H;

	// Extract first frame as reference
	cap >> frame;
	if (frame.size() != FINAL_SIZE) {
		downscale(frame, frame);
	}

	img_ref = frame.clone();

	Mat hsv, brightest, binary;
	vector<Mat> hsv_channels;

	cvtColor(img_ref, hsv, COLOR_BGR2HSV);
	split(hsv, hsv_channels);
	brightest = hsv_channels[2];
	cout << "HELLO";
	VideoWriter video;
	int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
	double fps = 10;
	video.open("E:/NIFA/datasets/grand_meadow_0001_all.avi", codec, fps, FINAL_SIZE, true);

	FileStorage fs(OMNI_MODEL_PATH, FileStorage::READ);
	Mat K, D, xi;
	fs["cameraMatrix"] >> K;
	fs["D"] >> D;
	fs["xi"] >> xi;

	Mat world_coords, rvec, tvec;
	vector<Point2f> pixel_coords, sampling_coords;
	vector<Point2d> new_pixel_coords;
	markFiducials(world_coords, pixel_coords);
	estimateCameraPose(world_coords, pixel_coords, K, D, rvec, tvec, new_pixel_coords);

	//// elgin
	//Point3f min_corner = Point3f(44.1285, -92.296, 0);
	//Point3f max_corner = Point3f(44.1303, -92.293, 0);

	//// becker
	//Point3f min_corner = Point3f(45.344873, -93.860312, 0);
	//Point3f max_corner = Point3f(45.347070, -93.857618, 0);

	// grand_meadow
	Point3f min_corner = Point3f(43.622596, -92.561403, 0);
	Point3f max_corner = Point3f(43.621269, -92.564, 0);

	Mat sampling_region = img_ref.clone();
	getSamplingRegion(sampling_region, min_corner, max_corner, Size(1000, 1400), K, D, rvec, tvec, true, sampling_coords);

	while (1) {
		cap >> frame;

		if (++count % 30 == 2 && count < 1000) {

			cout << "Processing frame #" << count << endl;

			if (frame.empty())
				break;

			if (frame.size() != FINAL_SIZE) {
				downscale(frame, frame);
			}

			getSurfRotation(img_ref, frame, H);

			//warpPerspective(frame, img_rot, H, img_ref.size());
			warpAffine(frame, img_rot, H, img_ref.size());
			
			//imshow("Unrectified", frame);
			//imshow("Rectified", img_rot);

			//cout << img_rot.size() << endl;
			//cout << brightest.size() << endl;
			//cout << binary.size() << endl;

			//if (img_rot.size() != FINAL_SIZE) {
			//	downscale(img_rot, img_rot);
			//}

			getBinaryCloudMask(img_rot, brightest, binary);
			medianBlur(binary, binary, 31);

			cvtColor(binary, binary, COLOR_GRAY2RGB);

			getRegisteredImage(img_rot, sampling_coords, false, img_rot);
			getRegisteredImage(binary, sampling_coords, false, binary);

			if (img_rot.size() != FINAL_SIZE) {
				downscale(img_rot, img_rot);
			}

			if (binary.size() != FINAL_SIZE) {
				downscale(binary, binary);
			}

			//imshow("Binary Cloud Mask", binary);

			//Mat v1, v2, v3, v4, row1, row2, canvas;
			//omnidir::undistortImage(img_ref, v1, K, D, xi, omnidir::RECTIFY_PERSPECTIVE);
			//omnidir::undistortImage(frame, v2, K, D, xi, omnidir::RECTIFY_PERSPECTIVE);
			//omnidir::undistortImage(img_rot, v3, K, D, xi, omnidir::RECTIFY_PERSPECTIVE);
			//omnidir::undistortImage(binary, v4, K, D, xi, omnidir::RECTIFY_PERSPECTIVE);

			//hconcat(v1, v2, row1);
			//cvtColor(v4, v4, COLOR_GRAY2RGB);
			//hconcat(v3, v4, row2);
			//vconcat(row1, row2, canvas);

			Mat row1, row2, canvas;
			hconcat(sampling_region, frame, row1);
			hconcat(img_rot, binary, row2);
			vconcat(row1, row2, canvas);
			if (canvas.size() != FINAL_SIZE) {
				downscale(canvas, canvas);
			}
			

			//pyrDown(canvas, canvas);
			//pyrDown(canvas, canvas);
			//resize(canvas, canvas, Size(canvas.cols / RESIZE_FACTOR, canvas.rows / RESIZE_FACTOR));
			//imshow("canvas", canvas);
			////cout << canvas.size << endl;
			imshow("wow", canvas);

			video << canvas;
		}
		else if(count >= 1000) {
			break;
		}

		//img_ref, frame, img_rot, binary

		char c = (char)waitKey(1);
		if (c == 27)
			break;
		
	}

	cap.release();
	cv::destroyAllWindows();

	//Mat img_ref, img_rot, new_rot, H;
	//img_ref = imread(ROT_ROTATED_PATH);
	//img_rot = imread(ROT_REF_PATH);

	//getSurfRotation(img_ref, img_rot, H);
	//warpAffine(img_rot, new_rot, H, img_ref.size());

	//Mat hsv, brightest, binary_old, binary_new;
	//vector<Mat> hsv_channels;

	//cvtColor(img_ref, hsv, COLOR_BGR2HSV);
	//split(hsv, hsv_channels);
	//brightest = hsv_channels[2];
	//getBinaryCloudMask(img_rot, brightest, binary_old);
	//getBinaryCloudMask(new_rot, brightest, binary_new);
	//medianBlur(binary_old, binary_old, 7);
	//medianBlur(binary_new, binary_new, 7);

	//Mat element = getStructuringElement(0, Size(5, 5), Point(0, 0));
	////morphologyEx(binary_old, binary_old, MORPH_OPEN, element);
	////morphologyEx(binary_new, binary_new, MORPH_OPEN, element);

	//while (1) {
	//	imshow("Reference", img_ref);
	//	imshow("Unrotated", img_rot);
	//	imshow("Rotated", new_rot);
	//	imshow("Old Binary Cloud Mask", binary_old);
	//	imshow("New Binary Cloud Mask", binary_new);
	//	waitKey(1);
	//}

	return 0;
}
//
//#include "opencv2/video/tracking.hpp"
//#include "opencv2/imgproc.hpp"
//#include "opencv2/videoio.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/opencv.hpp"
//
//#include <iostream>
//#include <fstream>
//#include <ctype.h>
//
//using namespace cv;
//using namespace std;
//
//const string GCP_PATH = "D:/NIFA/grand_meadow_07_08_2020/gcp_text.txt";
////const string VIDEO_PATH = "D:/NIFA/grand_meadow_07_08_2020/DJI_0002.MOV";
//const string VIDEO_PATH = "D:/NIFA/low_res.mp4";
//const string CAM_PATH = "0"; // Use for webcam/live input.
//
//Point2f point;
//bool clickStatus;
//int fiducial_index;
//
//// Read real-world coordinates for fiducial points from pre-formatted CSV.
//void read_fiducials_from_csv(string f, vector<string>& point_names, vector<Point2f>& point_coords) {
//    ifstream fin(f);
//    if (!fin.is_open()) throw runtime_error("Could not open file!");
//
//    vector<string> row;
//    string line, word, temp;
//
//    float x, y;
//
//    while (getline(fin, line)) {
//        row.clear();
//
//        stringstream s(line);
//
//        while (getline(s, word, ',')) {
//            row.push_back(word);
//        }
//
//        point_names.push_back(row[0]);
//
//        x = stof(row[1]);
//        y = stof(row[2]);
//        point_coords.push_back(Point2f(x, y));
//    }
//
//    fin.close();
//}
//
//// Mark fiducials read from CSV
//void mark_fiducials(VideoCapture& cap, Mat& image, vector<Point2f>(&points)[2], vector<string>& point_names, vector<Point2f>& point_coords) {
//    cout << "When the frame is correct, click to begin marking fiducials." << endl;
//    
//    while (!clickStatus) {
//        cap >> image;
//        
//        //resize(image, image, Size(384 * 2, 216 * 2));
//        imshow("Point Tracking", image);
//        waitKey(1);
//    }
//
//    fiducial_index = 0;
//    clickStatus = false;
//
//    cout << point_names.size() << " fiducial markers found. For each fiducial marker, click the corresponding location on the image." << endl;
//
//    if (point_names.size() == 0) {
//        cout << "No fiducial markers found. Check fiducials csv." << endl;
//        return;
//    }
//
//    for (int fd_idx = 0; fd_idx < 11; fd_idx++) {
//        while (!clickStatus) { waitKey(1); }
//        points[0].push_back(point);
//        clickStatus = false;
//        //cout << point_names[fd_idx] << ": " << point_coords[fd_idx].x << "," << point_coords[fd_idx].y << endl;
//    }
//}
//
//// Right click event handler
//static void onMouseClick(int event, int x, int y, int /*flags*/, void* /*param*/) {
//    if (event == EVENT_LBUTTONDOWN)
//    {
//        point = Point2f((float)x, (float)y);
//        clickStatus = true;
//    }
//}
//
//int main(int argc, char** argv) {
//    cout << CV_VERSION << endl;
//    VideoCapture cap;
//    TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
//    Size subPixWinSize(10, 10), winSize(31, 31);
//    const int MAX_COUNT = 500;
//    bool autoInit = false;
//
//    CommandLineParser parser(argc, argv, "{@input|0|}");
//    string input; // = parser.get<string>("@input");
//
//    input = VIDEO_PATH;
//    //cap.set(CAP_PROP_FOURCC, cv::VideoWriter::fourcc('D', 'I', 'V', '4'));
//
//    if (input.size() == 1 && isdigit(input[0]))
//        cap.open(input[0] - '0');
//    else
//        cap.open(input);
//    if (!cap.isOpened())
//    {
//        cout << "Could not initialize capturing...\n";
//        return 0;
//    }
//
//    //cap.set(CAP_PROP_AUTO_EXPOSURE, 1);
//    //cap.set(CAP_PROP_EXPOSURE, -5);
//
//    namedWindow("Point Tracking", 1);
//    setMouseCallback("Point Tracking", onMouseClick, 0);
//
//    Mat gray, prevGray, image;
//    vector<Point2f> points[2];
//
//    vector<string> point_names;
//    vector<Point2f> point_coords;
//
//    read_fiducials_from_csv(GCP_PATH, point_names, point_coords);
//    //mark_fiducials(cap, image, points, point_names, point_coords);
//
//    bool uninitialized = true;
//    vector<Point2f> orig_points;
//
//    int num_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
//    cout << num_frames << " frames." << endl;
//
//    Mat maxBrightness;
//    Mat diff;
//    Mat hsv;
//    vector<Mat> hsv_channels;
//
//    for (int i = 0; i < num_frames; i++) {
//        if (i++ % 5 != 0) {
//            cap >> image;
//            continue;
//        }
//        cap >> image;
//        GaussianBlur(image, image, Size(7, 7), 0);
//        if (image.empty())
//            break;       
//
//        //resize(image, image, Size(384 * 2, 216 * 2));
//
//
//        if (autoInit) {
//            // automatic initialization
//            goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
//            points[0] = points[1];
//            autoInit = false;
//        } else if (!points[0].empty()) {
//            vector<uchar> status;
//            vector<float> err;
//
//            if (prevGray.empty()) gray.copyTo(prevGray);
//            
//            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize, 3, termcrit, 0, 0.001);
//            cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
//            size_t i, k;
//            for (i = k = 0; i < points[1].size(); i++)
//            {
//                if (!status[i]) {
//                    cout << "lost track of point at" << points[0][i].x << ", " << points[0][i].y;
//                    //continue;
//                }
//
//                if (!uninitialized) orig_points[k] = orig_points[i];
//                points[1][k++] = points[1][i];
//                circle(image, points[1][i], 3, Scalar(0, 0, 255), -1, 8);
//            }
//            points[1].resize(k);
//        }
//
//        if (uninitialized) {
//            orig_points = points[1];
//            //uninitialized = false;
//        }
//
//        if (orig_points.size() == points[1].size() && orig_points.size() > 0) {
//            Mat H = estimateAffinePartial2D(points[1], orig_points);
//            try {
//                warpAffine(image, image, H, image.size());
//            } catch(cv::Exception& e) {
//                const char* err_msg = e.what();
//                cout << err_msg << endl;
//                cout << "Transformation matrix likely not the correct size or empty." << endl;
//            }
//        }
//
//        cvtColor(image, hsv, COLOR_BGR2HSV);
//        vector<Mat> hsv_channels;
//        split(hsv, hsv_channels);
//        Mat v = hsv_channels[2];
//        if (uninitialized) {
//            maxBrightness = v;
//            uninitialized = false;
//        }
//        //axBrightness = max(maxBrightness, v);
//        subtract(maxBrightness, v, diff);
//        Mat img_bw;
//        threshold(diff, img_bw, 0, 255, THRESH_BINARY | THRESH_OTSU);
//
//        namedWindow("HSV Value");
//        imshow("HSV Value", v);
//
//        namedWindow("Diff");
//        imshow("Diff", img_bw);
//
//        imshow("Point Tracking", image);
//        char c = (char) waitKey(1);
//
//        if (c == 27) break;
//        switch (c) {
//            case 'r':
//                autoInit = true;
//                break;
//            case 'c':
//                points[0].clear();
//                points[1].clear();
//                break;
//        }
//
//        swap(points[1], points[0]);
//        cv::swap(prevGray, gray);
//    }
//
//    cap.release();
//    destroyAllWindows();
//
//    return 0;
//}