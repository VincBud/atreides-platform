"""N-source fan-in: ordering, absence, heterogeneity, determinism, bounded state.

Every stream here is synthetic. Synthetic correctness is not real-world
robustness, and nothing in this file claims otherwise: it proves the schedule is
what the arithmetic says it is, on cameras nobody has to own.
"""

from __future__ import annotations

import pytest

from atreides_vision.errors import VenueSessionError
from atreides_vision.timing import StreamFrame, StreamRef, TimeUncertainty
from atreides_vision.venue.models import StreamAbsence, StreamActivity, VenueTick
from atreides_vision.venue.scheduler import StreamCursor, merge_streams
from venue_fakes import FakeStreamCursor, frame_image, stream_frames

CAM_A = StreamRef("cam_a", 0)
CAM_B = StreamRef("cam_b", 0)
CAM_C = StreamRef("cam_c", 0)


def cursor(
    stream: StreamRef,
    count: int,
    fps: float,
    offset_seconds: float = 0.0,
    timing: TimeUncertainty | None = None,
) -> FakeStreamCursor:
    return FakeStreamCursor(stream, stream_frames(stream, count, fps, offset_seconds, timing))


def activities(tick: VenueTick) -> dict[str, StreamActivity]:
    """Every stream in a tick, contributing or not, keyed by its string form."""
    states: dict[str, StreamActivity] = {
        str(contribution.stream): StreamActivity.CONTRIBUTED for contribution in tick.contributions
    }
    for absence in tick.absent:
        states[str(absence.stream)] = absence.activity
    return states


class TestSingleStream:
    def test_one_camera_produces_its_own_frames_in_order(self) -> None:
        ticks = list(merge_streams([cursor(CAM_A, 4, fps=25.0)]))

        assert [tick.sequence for tick in ticks] == [0, 1, 2, 3]
        assert [tick.venue_micros for tick in ticks] == [0, 40_000, 80_000, 120_000]
        assert all(len(tick.contributions) == 1 for tick in ticks)
        assert all(tick.absent == () for tick in ticks)

    def test_no_streams_at_all_yields_nothing(self) -> None:
        assert list(merge_streams([])) == []

    def test_a_stream_that_decoded_nothing_yields_no_ticks(self) -> None:
        assert list(merge_streams([cursor(CAM_A, 0, fps=25.0)])) == []


class TestHeterogeneousFrameRates:
    def test_25_and_30_fps_interleave_on_venue_time_not_frame_index(self) -> None:
        ticks = list(merge_streams([cursor(CAM_A, 6, fps=25.0), cursor(CAM_B, 6, fps=30.0)]))

        # 25 FPS lands on 0, 40, 80, 120, 160, 200 ms; 30 FPS on 0, 33.333,
        # 66.667, 100, 133.333, 166.667 ms. Only the shared instant at zero
        # coincides, so eleven distinct instants, not six.
        assert len(ticks) == 11
        assert [tick.venue_micros for tick in ticks] == sorted(tick.venue_micros for tick in ticks)
        assert len({tick.venue_micros for tick in ticks}) == len(ticks)
        assert len(ticks[0].contributions) == 2

        # Frame indices are never assumed to line up: at the second tick only
        # the 30 FPS stream has a frame, and its index differs from the other's.
        assert [str(contribution.stream) for contribution in ticks[1].contributions] == ["cam_b#0"]

    def test_a_third_non_commensurate_cadence_still_orders_cleanly(self) -> None:
        ticks = list(
            merge_streams(
                [
                    cursor(CAM_A, 5, fps=25.0),
                    cursor(CAM_B, 5, fps=30.0),
                    cursor(CAM_C, 5, fps=24000 / 1001),  # 23.976, the NTSC film rate
                ]
            )
        )

        keys = [tick.venue_micros for tick in ticks]
        assert keys == sorted(keys)
        assert len(set(keys)) == len(keys)
        assert sum(len(tick.contributions) for tick in ticks) == 15

    def test_every_contribution_shares_the_tick_instant_exactly(self) -> None:
        for tick in merge_streams([cursor(CAM_A, 8, fps=25.0), cursor(CAM_B, 8, fps=50.0)]):
            assert all(
                contribution.venue_micros == tick.venue_micros
                for contribution in tick.contributions
            )

    def test_a_faster_stream_leaves_the_slower_one_between_frames(self) -> None:
        ticks = list(merge_streams([cursor(CAM_A, 3, fps=25.0), cursor(CAM_B, 6, fps=50.0)]))

        assert activities(ticks[1]) == {
            "cam_a#0": StreamActivity.BETWEEN_FRAMES,
            "cam_b#0": StreamActivity.CONTRIBUTED,
        }


class TestStartOffsets:
    def test_a_stream_that_starts_later_is_not_yet_started(self) -> None:
        ticks = list(
            merge_streams(
                [cursor(CAM_A, 4, fps=25.0), cursor(CAM_B, 4, fps=25.0, offset_seconds=1.0)]
            )
        )

        assert activities(ticks[0]) == {
            "cam_a#0": StreamActivity.CONTRIBUTED,
            "cam_b#0": StreamActivity.NOT_YET_STARTED,
        }
        first_b = next(
            tick
            for tick in ticks
            if any(contribution.stream == CAM_B for contribution in tick.contributions)
        )
        assert first_b.venue_micros == 1_000_000

    def test_venue_time_reflects_the_offset_while_source_time_does_not(self) -> None:
        ticks = list(merge_streams([cursor(CAM_B, 2, fps=25.0, offset_seconds=7.5)]))

        first = ticks[0].contributions[0]
        assert first.source_seconds == 0.0
        assert first.venue_seconds == 7.5
        assert first.frame_index == 1


class TestDifferentDurations:
    def test_a_finished_stream_is_ended_while_the_others_continue(self) -> None:
        ticks = list(merge_streams([cursor(CAM_A, 2, fps=25.0), cursor(CAM_B, 5, fps=25.0)]))

        assert activities(ticks[0])["cam_a#0"] is StreamActivity.CONTRIBUTED
        assert activities(ticks[-1]) == {
            "cam_a#0": StreamActivity.ENDED,
            "cam_b#0": StreamActivity.CONTRIBUTED,
        }
        assert len(ticks) == 5

    def test_an_empty_stream_is_ended_from_the_very_first_tick(self) -> None:
        ticks = list(merge_streams([cursor(CAM_A, 3, fps=25.0), cursor(CAM_B, 0, fps=25.0)]))

        assert len(ticks) == 3
        assert all(activities(tick)["cam_b#0"] is StreamActivity.ENDED for tick in ticks)

    def test_all_three_absent_states_appear_in_one_session(self) -> None:
        ticks = list(
            merge_streams(
                [
                    cursor(CAM_A, 4, fps=50.0),
                    cursor(CAM_B, 1, fps=50.0),
                    cursor(CAM_C, 2, fps=50.0, offset_seconds=0.05),
                ]
            )
        )

        seen = {absence.activity for tick in ticks for absence in tick.absent}
        assert seen == {
            StreamActivity.BETWEEN_FRAMES,
            StreamActivity.NOT_YET_STARTED,
            StreamActivity.ENDED,
        }


class TestEpochs:
    def test_two_epochs_of_one_camera_never_collide(self) -> None:
        first, second = StreamRef("cam_a", 0), StreamRef("cam_a", 1)
        ticks = list(
            merge_streams(
                [
                    FakeStreamCursor(first, stream_frames(first, 2, fps=25.0)),
                    FakeStreamCursor(
                        second, stream_frames(second, 2, fps=25.0, offset_seconds=10.0)
                    ),
                ]
            )
        )

        contributing = [
            (str(contribution.stream), contribution.frame_index)
            for tick in ticks
            for contribution in tick.contributions
        ]
        assert contributing == [
            ("cam_a#0", 1),
            ("cam_a#0", 2),
            ("cam_a#1", 1),
            ("cam_a#1", 2),
        ]
        # The same camera, the same tracker numbering, two unrelated recordings.
        assert ticks[0].activity_of("cam_a") is StreamActivity.CONTRIBUTED
        assert ticks[0].absent[0].stream == second

    def test_epochs_order_after_the_camera_id(self) -> None:
        streams = [StreamRef("cam_a", 1), StreamRef("cam_a", 0), StreamRef("cam_b", 0)]
        cursors: list[StreamCursor] = [
            FakeStreamCursor(stream, stream_frames(stream, 1, fps=25.0)) for stream in streams
        ]

        tick = next(iter(merge_streams(cursors)))

        assert [str(contribution.stream) for contribution in tick.contributions] == [
            "cam_a#0",
            "cam_a#1",
            "cam_b#0",
        ]


class TestDeterminism:
    def test_cursor_order_does_not_reach_the_output(self) -> None:
        def walk(reverse: bool) -> list[tuple[int, int, tuple[str, ...], tuple[str, ...]]]:
            cursors: list[StreamCursor] = [
                cursor(CAM_A, 5, fps=25.0),
                cursor(CAM_B, 4, fps=30.0),
                cursor(CAM_C, 3, fps=25.0, offset_seconds=0.02),
            ]
            if reverse:
                cursors.reverse()
            return [
                (
                    tick.venue_micros,
                    tick.sequence,
                    tuple(str(item.stream) for item in tick.contributions),
                    tuple(f"{item.stream}:{item.activity.value}" for item in tick.absent),
                )
                for tick in merge_streams(cursors)
            ]

        assert walk(reverse=False) == walk(reverse=True)

    def test_two_identical_walks_produce_identical_ticks(self) -> None:
        def walk() -> list[VenueTick]:
            return list(merge_streams([cursor(CAM_A, 4, fps=25.0), cursor(CAM_B, 4, fps=30.0)]))

        # StreamFrame equality ignores pixels, so this compares the schedule.
        assert walk() == walk()

    def test_camera_ids_order_by_code_point(self) -> None:
        streams = [StreamRef(name, 0) for name in ("cam_B", "cam_a", "cam_A")]
        cursors: list[StreamCursor] = [
            FakeStreamCursor(stream, stream_frames(stream, 1, fps=25.0)) for stream in streams
        ]

        tick = next(iter(merge_streams(cursors)))

        assert [contribution.stream.camera_id for contribution in tick.contributions] == [
            "cam_A",
            "cam_B",
            "cam_a",
        ]


class TestBoundedState:
    def test_at_most_one_frame_is_held_per_stream(self) -> None:
        cursors = [cursor(CAM_A, 40, fps=25.0), cursor(CAM_B, 40, fps=30.0)]

        for _ in merge_streams(cursors):
            for held in cursors:
                in_flight = held.frames_materialised - held.frames_consumed
                assert 0 <= in_flight <= 1

        for held in cursors:
            assert held.frames_materialised == held.frames_consumed == 40

    def test_a_long_session_does_not_grow_per_tick_state(self) -> None:
        """Churn many frames; the merge must keep nothing but per-stream markers."""
        cursors = [cursor(CAM_A, 500, fps=25.0), cursor(CAM_B, 500, fps=30.0)]

        ticks = sum(1 for _ in merge_streams(cursors))

        assert ticks > 500
        assert all(held.frames_materialised == 500 for held in cursors)


class TestRefusals:
    def test_two_cursors_for_one_stream(self) -> None:
        with pytest.raises(VenueSessionError, match="same stream"):
            list(merge_streams([cursor(CAM_A, 2, fps=25.0), cursor(CAM_A, 2, fps=25.0)]))

    def test_a_cursor_offering_another_stream_s_frame(self) -> None:
        frames = stream_frames(CAM_B, 1, fps=25.0)
        with pytest.raises(VenueSessionError, match="provenance"):
            list(merge_streams([FakeStreamCursor(CAM_A, frames)]))

    def test_venue_time_that_does_not_advance(self) -> None:
        stalled = [
            StreamFrame(
                stream=CAM_A,
                frame_index=index,
                source_seconds=0.0,
                venue_seconds=0.0,
                timing=TimeUncertainty.unknown(),
                image=frame_image(index),
            )
            for index in (1, 2)
        ]
        with pytest.raises(VenueSessionError, match="must advance"):
            list(merge_streams([FakeStreamCursor(CAM_A, stalled)]))

    def test_a_failing_stream_ends_the_session(self) -> None:
        class ExplodingCursor:
            def __init__(self, stream: StreamRef) -> None:
                self._stream = stream
                self.peeks = 0

            @property
            def stream(self) -> StreamRef:
                return self._stream

            def peek(self) -> StreamFrame | None:
                self.peeks += 1
                if self.peeks > 1:
                    raise RuntimeError("decoder died")
                return stream_frames(self._stream, 1, fps=25.0)[0]

            def advance(self) -> None:
                return None

        with pytest.raises(RuntimeError, match="decoder died"):
            list(merge_streams([cursor(CAM_A, 3, fps=25.0), ExplodingCursor(CAM_B)]))


class TestTickInvariants:
    def test_a_tick_with_no_contributions_is_refused(self) -> None:
        with pytest.raises(VenueSessionError, match="at least one contributing"):
            VenueTick(venue_micros=0, sequence=0, contributions=())

    def test_a_contribution_at_another_instant_is_refused(self) -> None:
        frames = stream_frames(CAM_A, 2, fps=25.0)
        with pytest.raises(VenueSessionError, match="one instant"):
            VenueTick(venue_micros=0, sequence=0, contributions=tuple(frames))

    def test_one_stream_cannot_contribute_twice_to_a_tick(self) -> None:
        """Two frames from one stream at one instant would double that camera.

        The camera would look like one contributor to ``activity_of`` while
        supplying two frames, so the duplication is invisible at the only
        per-camera accessor. Rejected at construction instead.
        """
        frame = stream_frames(CAM_A, 1, fps=25.0)[0]

        with pytest.raises(VenueSessionError, match="more than once"):
            VenueTick(venue_micros=0, sequence=0, contributions=(frame, frame))

    def test_one_stream_cannot_be_absent_twice_from_a_tick(self) -> None:
        with pytest.raises(VenueSessionError, match="more than once"):
            VenueTick(
                venue_micros=0,
                sequence=0,
                contributions=tuple(stream_frames(CAM_A, 1, fps=25.0)),
                absent=(
                    StreamAbsence(CAM_B, StreamActivity.ENDED),
                    StreamAbsence(CAM_B, StreamActivity.ENDED),
                ),
            )

    def test_a_stream_cannot_be_present_and_absent(self) -> None:
        frames = stream_frames(CAM_A, 1, fps=25.0)
        with pytest.raises(VenueSessionError, match="both contributing and absent"):
            VenueTick(
                venue_micros=0,
                sequence=0,
                contributions=tuple(frames),
                absent=(StreamAbsence(CAM_A, StreamActivity.ENDED),),
            )

    def test_uncanonical_order_is_refused(self) -> None:
        with pytest.raises(VenueSessionError, match="canonical"):
            VenueTick(
                venue_micros=0,
                sequence=0,
                contributions=tuple(stream_frames(CAM_A, 1, fps=25.0)),
                absent=(
                    StreamAbsence(CAM_C, StreamActivity.ENDED),
                    StreamAbsence(CAM_B, StreamActivity.ENDED),
                ),
            )

    def test_an_absence_cannot_claim_to_have_contributed(self) -> None:
        with pytest.raises(VenueSessionError, match="did not contribute"):
            StreamAbsence(CAM_A, StreamActivity.CONTRIBUTED)

    def test_tick_seconds_are_quantised_while_contributions_are_not(self) -> None:
        precise = StreamFrame(
            stream=CAM_A,
            frame_index=1,
            source_seconds=1.000_000_4,
            venue_seconds=1.000_000_4,
            timing=TimeUncertainty.unknown(),
            image=frame_image(1),
        )
        tick = VenueTick(venue_micros=1_000_000, sequence=0, contributions=(precise,))

        assert tick.venue_seconds == 1.0
        assert tick.contributions[0].venue_seconds == 1.000_000_4

    def test_alignment_is_the_worst_of_the_contributors(self) -> None:
        ticks = list(
            merge_streams(
                [
                    cursor(CAM_A, 2, fps=25.0, timing=TimeUncertainty.declared(0.2)),
                    cursor(CAM_B, 2, fps=25.0, timing=TimeUncertainty.declared(0.9)),
                ]
            )
        )

        assert ticks[0].alignment == TimeUncertainty.declared(0.9)

    def test_an_unaligned_contributor_makes_the_whole_tick_unaligned(self) -> None:
        ticks = list(
            merge_streams(
                [
                    cursor(CAM_A, 2, fps=25.0, timing=TimeUncertainty.declared(0.2)),
                    cursor(CAM_B, 2, fps=25.0),
                ]
            )
        )

        assert not ticks[0].alignment.is_established

    def test_cameras_and_activity_lookups(self) -> None:
        ticks = list(merge_streams([cursor(CAM_A, 1, fps=25.0), cursor(CAM_B, 3, fps=25.0)]))

        assert ticks[0].cameras() == ("cam_a", "cam_b")
        assert ticks[1].cameras() == ("cam_b",)
        assert ticks[1].activity_of("cam_a") is StreamActivity.ENDED
        assert ticks[1].activity_of("cam_b") is StreamActivity.CONTRIBUTED
        assert ticks[1].activity_of("cam_unknown") is None


class TestActivityOfACameraWithSeveralRecordings:
    """A camera's epochs are aggregated, and a restart gap has a right answer."""

    FIRST = StreamRef("cam_a", 0)
    SECOND = StreamRef("cam_a", 1)

    def gap_tick(self) -> VenueTick:
        """Epoch 0 finished, epoch 1 not begun, another camera carrying the tick."""
        return VenueTick(
            venue_micros=0,
            sequence=0,
            contributions=tuple(stream_frames(CAM_B, 1, fps=25.0)),
            absent=(
                StreamAbsence(self.FIRST, StreamActivity.ENDED),
                StreamAbsence(self.SECOND, StreamActivity.NOT_YET_STARTED),
                StreamAbsence(CAM_C, StreamActivity.ENDED),
            ),
        )

    def test_a_restart_gap_reads_as_between_frames(self) -> None:
        """The camera has demonstrably started and will demonstrably return."""
        assert self.gap_tick().activity_of("cam_a") is StreamActivity.BETWEEN_FRAMES

    def test_a_camera_that_is_wholly_finished_still_reads_as_ended(self) -> None:
        assert self.gap_tick().activity_of("cam_c") is StreamActivity.ENDED

    def test_per_stream_detail_survives_the_aggregation(self) -> None:
        """The two epochs stay separately visible; only the summary merges them."""
        absences = {str(absence.stream): absence.activity for absence in self.gap_tick().absent}

        assert absences["cam_a#0"] is StreamActivity.ENDED
        assert absences["cam_a#1"] is StreamActivity.NOT_YET_STARTED

    def test_a_camera_whose_recordings_are_all_still_ahead_has_not_started(self) -> None:
        tick = VenueTick(
            venue_micros=0,
            sequence=0,
            contributions=tuple(stream_frames(CAM_B, 1, fps=25.0)),
            absent=(
                StreamAbsence(self.FIRST, StreamActivity.NOT_YET_STARTED),
                StreamAbsence(self.SECOND, StreamActivity.NOT_YET_STARTED),
            ),
        )

        assert tick.activity_of("cam_a") is StreamActivity.NOT_YET_STARTED

    def test_a_contributing_epoch_wins_over_every_other_state(self) -> None:
        tick = VenueTick(
            venue_micros=0,
            sequence=0,
            contributions=tuple(stream_frames(self.SECOND, 1, fps=25.0)),
            absent=(StreamAbsence(self.FIRST, StreamActivity.ENDED),),
        )

        assert tick.activity_of("cam_a") is StreamActivity.CONTRIBUTED
