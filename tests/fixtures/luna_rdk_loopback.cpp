// Minimal RDK-side fixture for the native Gento hardware gateway integration test.
#include <apex_ros_wire.h>
#include <dds/dds.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>

namespace {
dds_entity_t checked(dds_entity_t entity) {
    if (entity < 0) {
        throw std::runtime_error(dds_strretcode(entity));
    }
    return entity;
}

void run(dds_domainid_t domain) {
    using namespace std::chrono_literals;
    const auto participant = checked(dds_create_participant(domain, nullptr, nullptr));
    const auto commands = checked(dds_create_topic(participant,
        &geometry_msgs_msg_dds__TwistStamped__desc, "rt/move/ManualMoveCmd", nullptr, nullptr));
    const auto odometry = checked(dds_create_topic(participant,
        &nav_msgs_msg_dds__Odometry__desc, "rt/controller/odom", nullptr, nullptr));
    auto *qos = dds_create_qos();
    dds_qset_history(qos, DDS_HISTORY_KEEP_LAST, 1);
    dds_qset_reliability(qos, DDS_RELIABILITY_RELIABLE, DDS_MSECS(50));
    const auto reader = checked(dds_create_reader(participant, commands, qos, nullptr));
    dds_qset_reliability(qos, DDS_RELIABILITY_BEST_EFFORT, 0);
    const auto writer = checked(dds_create_writer(participant, odometry, qos, nullptr));
    dds_delete_qos(qos);

    char world[] = "odom";
    char body[] = "base_link";
    nav_msgs_msg_dds__Odometry_ feedback{};
    feedback.header.frame_id = world;
    feedback.child_frame_id = body;
    feedback.pose.pose.orientation.w = 1.;
    for (;;) {
        void* sample{};
        dds_sample_info_t info{};
        if (const auto count = checked(dds_take(reader, &sample, &info, 1, 1)); count) {
            if (info.valid_data) {
                const auto &command = *static_cast<geometry_msgs_msg_dds__TwistStamped_ *>(sample);
                feedback.twist.twist = command.twist;
            }
            checked(dds_return_loan(reader, &sample, count));
        }
        const auto now = dds_time();
        feedback.header.stamp.sec = now / 1'000'000'000;
        feedback.header.stamp.nanosec = now % 1'000'000'000;
        checked(dds_write(writer, &feedback));
        std::this_thread::sleep_for(20ms);
    }
}
}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        return 2;
    }
    try {
        run(std::strtoul(argv[1], nullptr, 10));
    } catch (const std::exception &error) {
        std::fprintf(stderr, "RDK loopback fixture: %s\n", error.what());
        return 1;
    }
}
