from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import expression

from typing import Dict, Optional, Iterable, List, Set
from enum import Enum

from .fishers import fisher_significant
import re

_junit_table_engine = None
_table = None


def junit_table():
    global _junit_table_engine, _table
    if _table is None:
        _junit_table_engine = create_engine('bigquery://openshift-gce-devel')
        _table = Table('ci_analysis_us.junit', MetaData(bind=_junit_table_engine), autoload=True)
    return _table


# https://docs.sqlalchemy.org/en/14/orm/declarative_styles.html
Base = declarative_base()


DATETIME_INPUT_PATTERN = re.compile(r'(\d\d\d\d-\d\d-\d\d)[T: ]?(\d\d:\d\d).*')


class Junit(Base):
    # https://docs.sqlalchemy.org/en/14/orm/mapping_columns.html
    __table__ = junit_table()
    __mapper_args__ = {'primary_key': [__table__.c.schema_level, __table__.c.file_path, __table__.c.test_id, __table__.c.success_val]}

    schema_level = __table__.c.schema_level
    prowjob_build_id = __table__.c.prowjob_build_id
    file_path = __table__.c.file_path
    test_name = __table__.c.test_name
    duration_ms = __table__.c.duration_ms
    success = __table__.c.success
    skipped = __table__.c.skipped
    modified_time = __table__.c.modified_time
    test_id = __table__.c.test_id
    branch = __table__.c.branch
    prowjob_name = __table__.c.prowjob_name
    success_val = __table__.c.success_val
    network = __table__.c.network
    platform = __table__.c.platform
    arch = __table__.c.arch
    upgrade = __table__.c.upgrade
    variants = __table__.c.variants
    flake_count = __table__.c.flake_count

    @classmethod
    def platform_drill_key(cls) -> func.GenericFunction:
        return func.concat(cls.network, ' ', cls.upgrade, ' ', cls.platform, ' ', cls.arch)

    @classmethod
    def format_modified_time(cls, s: str):
        m = DATETIME_INPUT_PATTERN.match(s)
        if not m:
            raise ValueError(f'Invalid datetime format: {s}')
        return func.datetime(func.timestamp(m.group(1) + ' ' + m.group(2) + ':00+00'))


select = expression.select
any_value = func.any_value
count = func.count
sum = func.sum

EnvironmentName = str
ComponentName = str
CapabilityName = str
TestId = str
TestName = str


class TestRecordAssessment(Enum):
    EXTREME_REGRESSION = 'fire.png'
    SIGNIFICANT_REGRESSION = 'red.png'
    MISSING_IN_SAMPLE = 'red.png'
    NOT_SIGNIFICANT = 'green.png'
    MISSING_IN_BASIS = 'green.png'
    SIGNIFICANT_IMPROVEMENT = 'green-heart.png'


class TestRecord:

    @classmethod
    def aggregate_assessment(cls, test_assessments: Iterable[TestRecordAssessment]):
        overall_assessment: TestRecordAssessment = TestRecordAssessment.NOT_SIGNIFICANT
        found_missing_in_sample = False
        for test_assessment in test_assessments:
            # A single significant regression in any test means an overall regression
            if test_assessment is TestRecordAssessment.SIGNIFICANT_REGRESSION:
                return test_assessment

            if test_assessment is TestRecordAssessment.MISSING_IN_SAMPLE:
                # Prefer to return something more interesting like SIGNIFICANT_REGRESSION,
                # but if any sample is missing, we should technically show a regression
                # indicator in case someone nerfed a test by removing it.
                found_missing_in_sample = True

            if test_assessment is TestRecordAssessment.SIGNIFICANT_IMPROVEMENT:
                overall_assessment = test_assessment

            if overall_assessment is None or overall_assessment is TestRecordAssessment.NOT_SIGNIFICANT:
                overall_assessment = test_assessment

        if found_missing_in_sample:
            return TestRecordAssessment.MISSING_IN_SAMPLE

        return overall_assessment

    @classmethod
    def aggregate_test_record_assessment(cls, test_records: Iterable["TestRecord"]) -> TestRecordAssessment:
        return TestRecord.aggregate_assessment([test_record._assessment for test_record in test_records])

    def __init__(self,
                 environment_name: EnvironmentName,
                 test_id: TestId,
                 test_name: TestName,
                 success_count: int = 0,
                 failure_count: int = 0,
                 total_count: int = 0,
                 flake_count: int = 0,
                 upgrade: str = None,
                 assessment: TestRecordAssessment = None):
        self.environment_name = environment_name
        self.test_id = test_id
        self.test_name = test_name
        self.success_count = success_count
        self.failure_count = failure_count
        self.total_count = total_count
        self.flake_count = flake_count
        self.upgrade = upgrade
        self._assessment = assessment

    def assessment(self) -> TestRecordAssessment:
        if self._assessment is None:
            raise IOError(f'Could not find computed assessment for test_id {self.test_id}')
        return self._assessment

    def compute_assessment(self, basis_test_record: "TestRecord"):
        if self._assessment:
            # Already assessed
            return

        assessment: TestRecordAssessment = TestRecordAssessment.MISSING_IN_BASIS
        if basis_test_record and basis_test_record.total_count > 0:

            if basis_test_record.total_count > 0 and self.total_count == 0:
                return TestRecordAssessment.MISSING_IN_SAMPLE

            basis_pass_percentage = basis_test_record.success_count / basis_test_record.total_count
            sample_pass_percentage = self.success_count / self.total_count
            improved = sample_pass_percentage >= basis_pass_percentage

            significant = fisher_significant(
                self.failure_count,
                self.success_count,
                basis_test_record.failure_count,
                basis_test_record.success_count,
            )

            if significant:
                if improved:
                    assessment = TestRecordAssessment.SIGNIFICANT_IMPROVEMENT
                else:
                    if basis_pass_percentage - sample_pass_percentage > 0.15:
                        assessment = TestRecordAssessment.EXTREME_REGRESSION
                    else:
                        assessment = TestRecordAssessment.SIGNIFICANT_REGRESSION
            else:
                assessment = TestRecordAssessment.NOT_SIGNIFICANT

        self._assessment = assessment


class CapabilityTestRecords:

    def __init__(self, name: CapabilityName):
        self.name: ComponentName = name
        self.test_records: Dict[TestId, TestRecord] = dict()

    def get_test_ids(self):
        return self.test_records.keys()

    def add_test_record(self, test_record: TestRecord):
        if test_record.test_id in self.test_records:
            return
        self.test_records[test_record.test_id] = test_record

    def get_test_record(self, test_id: TestId) -> TestRecord:
        if test_id not in self.test_records:
            self.test_records[test_id] = TestRecord(
                test_id=test_id,
                test_name='',
                environment_name='n/a',
            )
        return self.test_records[test_id]

    def assessment(self) -> TestRecordAssessment:
        return TestRecord.aggregate_test_record_assessment(self.test_records.values())

    def build_mass_assessment_cache(self, basis_capability_test_records: "CapabilityTestRecords"):
        basis_test_ids = set(basis_capability_test_records.get_test_ids())
        sample_test_ids = set(self.get_test_ids())
        test_ids_not_in_basis = basis_test_ids.difference(sample_test_ids)

        all_ids = basis_test_ids.union(sample_test_ids)

        for test_id in test_ids_not_in_basis:
            basis_test_record = basis_capability_test_records.get_test_record(test_id)
            # TODO: if a test_id has been officially deprecated by staff, do not add this record
            place_holder_record = TestRecord(
                assessment=TestRecordAssessment.MISSING_IN_BASIS,
                # Match basis record elements so that the sample record
                # is sorted into the same component -> capability
                environment_name=basis_test_record.environment_name,
                test_id=basis_test_record.test_id,
                test_name=basis_test_record.test_name,
                upgrade=basis_test_record.upgrade,
            )
            self.add_test_record(place_holder_record)

        for test_id in all_ids:
            sample_test_record = self.test_records[test_id]
            if test_id in basis_capability_test_records.test_records:
                basis_test_record = basis_capability_test_records.test_records[test_id]
                sample_test_record.compute_assessment(basis_test_record)
            else:
                sample_test_record._assessment = TestRecordAssessment.MISSING_IN_BASIS


class ComponentTestRecords:

    def __init__(self, name: ComponentName, grouping_by_upgrade: bool):
        self.name: ComponentName = name
        self.grouping_by_upgrade = grouping_by_upgrade
        self.capability_test_records: Dict[CapabilityName, CapabilityTestRecords] = dict()

    def get_capability_names(self) -> Iterable[CapabilityName]:
        return self.capability_test_records.keys()

    def get_capability_test_records(self, capability_name: CapabilityName) -> CapabilityTestRecords:
        if capability_name in self.capability_test_records:
            return self.capability_test_records[capability_name]
        new_ctr = CapabilityTestRecords(capability_name)
        self.capability_test_records[capability_name] = new_ctr
        return new_ctr

    def find_associated_capabilities(self, test_record: TestRecord) -> Iterable[CapabilityName]:
        associated_capabilities: List[CapabilityName] = list()
        associated_capabilities.append(test_record.test_id[:1])  # Just a stop gap in place of looking these up from a database
        if not self.grouping_by_upgrade:
            if test_record.upgrade == 'upgrade-minor':
                associated_capabilities.append('y-stream upgrade')
            if test_record.upgrade == 'upgrade-micro':
                associated_capabilities.append('z-stream upgrade')
        return associated_capabilities

    def add_test_record(self, test_record: TestRecord):
        # Find all capabilities with which the test should be associated.
        for capability_name in self.find_associated_capabilities(test_record):
            self.get_capability_test_records(capability_name).add_test_record(test_record)

    def assessment(self) -> TestRecordAssessment:
        return TestRecord.aggregate_assessment([ctr.assessment() for ctr in self.capability_test_records.values()])

    def build_mass_assessment_cache(self, basis_component_test_records: "ComponentTestRecords"):
        all_names = set(self.get_capability_names())
        all_names.update(basis_component_test_records.get_capability_names())

        for name in all_names:
            sample_capability_test_records = self.get_capability_test_records(name)
            basis_capability_test_records = basis_component_test_records.get_capability_test_records(name)
            sample_capability_test_records.build_mass_assessment_cache(basis_capability_test_records)


class EnvironmentTestRecords:

    COMPONENT_NAME_PATTERN = re.compile(r'\[[^\\]+?\]')

    def __init__(self, name: EnvironmentName, grouping_by_upgrade=False):
        self.all_test_record_ids: Set[TestId] = set()
        self.name: EnvironmentName = name
        self.grouping_by_upgrade = grouping_by_upgrade
        self.component_test_records: Dict[ComponentName, ComponentTestRecords] = dict()

    def get_component_test_records(self, component_name: ComponentName) -> ComponentTestRecords:
        if component_name in self.component_test_records:
            return self.component_test_records[component_name]
        new_ctr = ComponentTestRecords(component_name, self.grouping_by_upgrade)
        self.component_test_records[component_name] = new_ctr
        return new_ctr

    def find_associated_components(self, test_record: TestRecord) -> Iterable[ComponentName]:
        return re.findall(EnvironmentTestRecords.COMPONENT_NAME_PATTERN, test_record.test_name)

    def add_test_record(self, test_record: TestRecord):
        self.all_test_record_ids.add(test_record.test_id)
        # Find a list of all component names to which this test belongs.
        # Register the test with each component name.
        for component_name in self.find_associated_components(test_record):
            component_test_records = self.get_component_test_records(component_name)
            component_test_records.add_test_record(test_record)

    def get_component_names(self):
        return self.component_test_records.keys()

    def assessment(self) -> TestRecordAssessment:
        return TestRecord.aggregate_assessment([ctr.assessment() for ctr in self.component_test_records.values()])

    def build_mass_assessment_cache(self, basis_environment_test_records: "EnvironmentTestRecords"):

        all_names = set(self.get_component_names())
        all_names.update(basis_environment_test_records.get_component_names())

        for name in all_names:
            sample_component_test_records = self.get_component_test_records(name)
            basis_component_test_records = basis_environment_test_records.get_component_test_records(name)
            sample_component_test_records.build_mass_assessment_cache(basis_component_test_records)


class EnvironmentModel:

    def get_environment_names(self):
        return self.environment_test_records.keys()

    def get_environment_test_records(self, env_name) -> EnvironmentTestRecords:
        if env_name in self.environment_test_records:
            return self.environment_test_records[env_name]
        new_etr = EnvironmentTestRecords(env_name)
        self.environment_test_records[env_name] = new_etr
        return new_etr

    def __init__(self):
        self.environment_test_records: Dict[EnvironmentName, EnvironmentTestRecords] = dict()

    def add_test_record(self, test_record: TestRecord):
        environment_test_records = self.get_environment_test_records(test_record.environment_name)
        environment_test_records.add_test_record(test_record)

    def read_in_query(self, query, group_by: str):
        group_by_elements = group_by.split(',')
        group_by_network = 'network' in group_by_elements
        group_by_upgrade = 'upgrade' in group_by_elements
        group_by_arch = 'arch' in group_by_elements
        group_by_platform = 'cloud' in group_by_elements or 'platform' in group_by_elements

        if not any((group_by_network, group_by_upgrade, group_by_arch, group_by_platform)):
            raise IOError('Invalid group-by values')

        for row in query.execute():
            env_name_components: List[str] = list()
            if group_by_platform:
                env_name_components.append(row.platform)
            if group_by_arch:
                env_name_components.append(row.arch)
            if group_by_network:
                env_name_components.append(row.network)
            if group_by_upgrade:
                env_name_components.append(row.upgrade)

            env_name: EnvironmentName = ' '.join(env_name_components)
            flake_count = row.flake_count
            success_count = row.success_count
            # In rare circumstances, based on the date range selected, it is possible for a failed test run to not be included
            # in the query while the success run (including a flake_count=1 reflecting the preceding, but un-selected
            # failure) is included. This could make total_count - flake_count a negative value.
            total_count = max(success_count, row.total_count - flake_count)

            r = TestRecord(
                environment_name=env_name,
                test_id=row.test_id,
                test_name=row.test_name,
                success_count=success_count,
                failure_count=total_count - success_count,
                total_count=total_count,
                flake_count=flake_count,
                upgrade=row.upgrade,
            )

            self.add_test_record(r)

    def build_mass_assessment_cache(self, basis_model: "EnvironmentModel"):

        # Get a list of all environment - including both basis and sample in case
        # there is one in basis that was nerfed.
        all_names = set(self.get_environment_names())
        all_names.update(basis_model.get_environment_names())

        for environment_name in all_names:
            sample_environment_test_records = self.get_environment_test_records(environment_name)
            basis_environment_test_records = basis_model.get_environment_test_records(environment_name)
            sample_environment_test_records.build_mass_assessment_cache(basis_environment_test_records)
