#include "Tensor/Quat.h"
#include "Common/Test.h"

//ehhh ?
static_assert(Tensor::is_tensor_v<Tensor::quatf>);

constexpr float epsilon = 1e-6;

//#define TEST_QUAT_EQ(a,b) TEST_EQ_EPS(Tensor::distance(a,b), 0, epsilon)
#define TEST_QUAT_EQ(a,b)\
	{\
		std::ostringstream ss;\
		float dist = (a).distance(b);\
		ss << __FILE__ << ":" << __LINE__ << ": " << #a << " == " << #b << " :: " << (a) << " == " << (b);\
		ss << " (distance=" << dist << ", epsilon=" << epsilon << ")";\
		std::string msg = ss.str();\
		if (dist > epsilon) {\
			msg += " FAILED!";\
			std::cout << msg << std::endl;\
			throw Common::Exception() << msg;\
		}\
		std::cout << msg << std::endl;\
	}

void test_Quat() {
	{
		//verify identity
		Tensor::quatf q;
		TEST_EQ(q.x, 0);
		TEST_EQ(q.y, 0);
		TEST_EQ(q.z, 0);
		TEST_EQ(q.w, 1);
	}

	{
		//real quaternion
		Tensor::quatf q = {0,0,0,2};
		
		//normal = unit
		TEST_EQ(normalize(q), Tensor::quatf());
		
		//real quaternion, conjugate = self
		TEST_EQ(q.conjugate(), q);
	}

	{
		//90' rotation
		float const sqrt_1_2 = sqrt(.5);
		auto rx = Tensor::quatf(1,0,0,.5*M_PI).fromAngleAxis();
		auto ry = Tensor::quatf(0,1,0,.5*M_PI).fromAngleAxis();
		auto rz = Tensor::quatf(0,0,1,.5*M_PI).fromAngleAxis();
		TEST_QUAT_EQ(rx, Tensor::quatf(sqrt_1_2, 0, 0, sqrt_1_2));
		TEST_QUAT_EQ(ry, Tensor::quatf(0, sqrt_1_2, 0, sqrt_1_2));
		TEST_QUAT_EQ(rz, Tensor::quatf(0, 0, sqrt_1_2, sqrt_1_2));
		
		//verify conjugate by components
		TEST_QUAT_EQ(rx.conjugate(), Tensor::quatf(-sqrt_1_2, 0, 0, sqrt_1_2));
		TEST_QUAT_EQ(ry.conjugate(), Tensor::quatf(0, -sqrt_1_2, 0, sqrt_1_2));
		TEST_QUAT_EQ(rz.conjugate(), Tensor::quatf(0, 0, -sqrt_1_2, sqrt_1_2));

		//verify negative by components
		TEST_QUAT_EQ(-rx, Tensor::quatf(-sqrt_1_2, 0, 0, -sqrt_1_2));
		TEST_QUAT_EQ(-ry, Tensor::quatf(0, -sqrt_1_2, 0, -sqrt_1_2));
		TEST_QUAT_EQ(-rz, Tensor::quatf(0, 0, -sqrt_1_2, -sqrt_1_2));

		// 180 rotation around each axis
		//180' rotation = w==0 = "pure quaternion"
		auto r2x = Tensor::quatf(1,0,0,M_PI).fromAngleAxis();
		auto r2y = Tensor::quatf(0,1,0,M_PI).fromAngleAxis();
		auto r2z = Tensor::quatf(0,0,1,M_PI).fromAngleAxis();
		TEST_QUAT_EQ(r2x, Tensor::quatf(1,0,0,0));
		TEST_QUAT_EQ(r2y, Tensor::quatf(0,1,0,0));
		TEST_QUAT_EQ(r2z, Tensor::quatf(0,0,1,0));
		
		//verify conjugate
		TEST_QUAT_EQ(r2x.conjugate(), Tensor::quatf(-1,0,0,0));
		TEST_QUAT_EQ(r2y.conjugate(), Tensor::quatf(0,-1,0,0));
		TEST_QUAT_EQ(r2z.conjugate(), Tensor::quatf(0,0,-1,0));

		//pure quaternion = negative equal to its conjugate
		TEST_QUAT_EQ(r2x.conjugate(), -r2x);
		TEST_QUAT_EQ(r2y.conjugate(), -r2y);
		TEST_QUAT_EQ(r2z.conjugate(), -r2z);
		
		//verify two 90' rotation around an axis == 180' rotation around the axis
		TEST_QUAT_EQ(rx*rx, r2x);
		TEST_QUAT_EQ(ry*ry, r2y);
		TEST_QUAT_EQ(rz*rz, r2z);

		//verify rx*ry, ry*rz, rz*rx
		TEST_QUAT_EQ(rx*ry, Tensor::quatf(.5f, .5f, .5f, .5f));
		TEST_QUAT_EQ(ry*rz, Tensor::quatf(.5f, .5f, .5f, .5f));
		TEST_QUAT_EQ(rz*rx, Tensor::quatf(.5f, .5f, .5f, .5f));

		// three rotations around each axis
		auto r3x = r2x*rx;
		auto r3y = r2y*ry;
		auto r3z = r2z*rz;
		
		//test associativity
		TEST_QUAT_EQ(r3x, rx*r2x);
		TEST_QUAT_EQ(r3y, ry*r2y);
		TEST_QUAT_EQ(r3z, rz*r2z);
	
		//verify three rotations is rotationally equivalent (up to negative) of a conjugate (equals inverse when normalized) rotation
		TEST_QUAT_EQ(-rx, r3x.conjugate());
		TEST_QUAT_EQ(-ry, r3y.conjugate());
		TEST_QUAT_EQ(-rz, r3z.conjugate());
	
		//quaterion question: how come this one has a sign flip on the axis while the reverse order doesn't?
		// hmm because the reverse-order , all the same, demonstrates the 2nd transform not affecting the 1st transform, while in this order the transforms do affect one another
		TEST_QUAT_EQ(rz*ry, Tensor::quatf(-.5f, .5f, .5f, .5f));
		TEST_QUAT_EQ(ry*rx, Tensor::quatf(.5f, .5f, -.5f, .5f));
		TEST_QUAT_EQ(rx*rz, Tensor::quatf(.5f, -.5f, .5f, .5f));
	
		TEST_QUAT_EQ(ry*rx*rz, rx);
		TEST_QUAT_EQ(rz*ry*rx, ry);
		TEST_QUAT_EQ(rx*rz*ry, rz);
		
		TEST_QUAT_EQ(rz*rx*ry, Tensor::quatf(0, sqrt_1_2, sqrt_1_2, 0));
		TEST_QUAT_EQ(rx*ry*rz, Tensor::quatf(sqrt_1_2, 0, sqrt_1_2, 0));
		TEST_QUAT_EQ(ry*rz*rx, Tensor::quatf(sqrt_1_2, sqrt_1_2, 0, 0));
	
		{
			Tensor::quatf::vec3 v = {1,2,3};
			auto vx = rx.rotate(v);
			TEST_QUAT_EQ(vx, Tensor::quatf::vec3(1, -3, 2));
			auto vy = ry.rotate(v);
			TEST_QUAT_EQ(vy, Tensor::quatf::vec3(3, 2, -1));
			auto vz = rz.rotate(v);
			TEST_QUAT_EQ(vz, Tensor::quatf::vec3(-2, 1, 3));
		}
	}
}
