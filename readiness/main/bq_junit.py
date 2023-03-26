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
TestUUID = str
TestId = str
TestName = str


class TestRecordAssessment(Enum):
    EXTREME_REGRESSION = (-3, 'Extreme regression', 'fire.png')
    SIGNIFICANT_REGRESSION = (-2, 'Significant regression', 'red.png')
    MISSING_IN_SAMPLE = (-1, 'No test runs in sample', 'red-question-mark.png')
    NOT_SIGNIFICANT = (0, 'No significant deviation', 'green.png')
    MISSING_IN_BASIS = (1, 'No records in basis data', 'green.png')
    SIGNIFICANT_IMPROVEMENT = (2, 'Significant improvement', 'green-heart.png')

    def __init__(self, val: int, description: str, image_path: str):
        self.val = val
        self.description = description
        self.image_path = image_path


class TestRecord:

    @classmethod
    def aggregate_assessment(cls, test_assessments: Iterable[TestRecordAssessment]):
        overall_assessment: TestRecordAssessment = TestRecordAssessment.NOT_SIGNIFICANT
        found_missing_in_sample = False
        for test_assessment in test_assessments:
            # A single significant regression in any test means an overall regression
            if test_assessment.val < 0:
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
        return TestRecord.aggregate_assessment([test_record.cached_assessment for test_record in test_records])

    def __init__(self,
                 platform: str,
                 network: str,
                 arch: str,
                 upgrade: str,
                 test_id: TestId,
                 test_name: TestName,
                 success_count: int = 0,
                 failure_count: int = 0,
                 total_count_minus_flakes: int = 0,
                 flake_count: int = 0,
                 assessment: TestRecordAssessment = None):
        self.test_id = test_id
        self.test_name = test_name
        self.success_count = success_count
        self.failure_count = failure_count
        self.total_count_minus_flakes = total_count_minus_flakes
        self.flake_count = flake_count
        self.cached_assessment = assessment
        self.success_rate = '{:.2f}'.format(0.0 if total_count_minus_flakes == 0 else 100 * success_count / total_count_minus_flakes)
        self.platform = platform
        self.network = network
        self.arch = arch
        self.upgrade = upgrade
        self.test_uuid: TestUUID = f'p={platform};n={network};a={arch};u={upgrade};id={test_id}'

    def assessment(self) -> TestRecordAssessment:
        if self.cached_assessment is None:
            return TestRecordAssessment.MISSING_IN_BASIS
        return self.cached_assessment

    def compute_assessment(self, basis_test_record: "TestRecord", alpha: float, regression_when_missing: bool):
        if self.cached_assessment:
            # Already assessed
            return

        assessment: TestRecordAssessment = TestRecordAssessment.MISSING_IN_BASIS
        if basis_test_record and basis_test_record.total_count_minus_flakes > 0:

            if basis_test_record.total_count_minus_flakes > 0 and self.total_count_minus_flakes == 0:
                if regression_when_missing:
                    return TestRecordAssessment.MISSING_IN_SAMPLE
                else:
                    return TestRecordAssessment.NOT_SIGNIFICANT

            basis_pass_percentage = basis_test_record.success_count / basis_test_record.total_count_minus_flakes
            sample_pass_percentage = self.success_count / self.total_count_minus_flakes
            improved = sample_pass_percentage >= basis_pass_percentage

            significant = fisher_significant(
                self.failure_count,
                self.success_count,
                basis_test_record.failure_count,
                basis_test_record.success_count,
                alpha=alpha,
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

        self.cached_assessment = assessment


class TestRecordSet:

    def __init__(self, test_id: TestId):
        self.test_id = test_id
        self.test_records: Dict[TestUUID, TestRecord] = dict()
        self.canonical_test_name = None

    def get_test_record_uuids(self) -> List[TestUUID]:
        return list(self.test_records.keys())

    def get_test_record(self, test_id: TestId) -> TestRecord:
        return self.test_records.get(test_id, None)

    def get_test_records(self) -> List[TestRecord]:
        return list(self.test_records.values())

    def add_test_record(self, test_record: TestRecord):
        if test_record.test_uuid in self.test_records:
            return
        self.test_records[test_record.test_uuid] = test_record
        if not self.canonical_test_name:
            # TODO: for now, just pick the first name we find and treat it as the canonical name.
            self.canonical_test_name = test_record.test_name

    def assessment(self) -> TestRecordAssessment:
        return TestRecord.aggregate_test_record_assessment(self.test_records.values())

    def get_test_name(self) -> str:
        # Over time, test names may change between releases. Return the canonical
        # for the test_id.
        return self.canonical_test_name


class CapabilityTestRecords:

    def __init__(self, name: CapabilityName):
        self.name: ComponentName = name
        self.test_record_sets: Dict[TestId, TestRecordSet] = dict()

    def get_test_record_set_ids(self):
        return list(self.test_record_sets.keys())

    def get_test_record_set(self, test_id: TestId):
        if test_id in self.test_record_sets:
            return self.test_record_sets[test_id]
        ntrs = TestRecordSet(test_id)
        self.test_record_sets[test_id] = ntrs
        return ntrs

    def get_test_record_sets(self):
        return self.test_record_sets.values()

    def add_test_record(self, test_record: TestRecord):
        self.get_test_record_set(test_record.test_id).add_test_record(test_record)

    def assessment(self) -> TestRecordAssessment:
        return TestRecord.aggregate_assessment([ctr.assessment() for ctr in self.test_record_sets.values()])


class ComponentTestRecords:

    def __init__(self, name: ComponentName, grouping_by_upgrade: bool):
        self.name: ComponentName = name
        self.grouping_by_upgrade = grouping_by_upgrade
        self.capability_test_records: Dict[CapabilityName, CapabilityTestRecords] = dict()

    def get_capability_names(self) -> Iterable[CapabilityName]:
        return list(self.capability_test_records.keys())

    def get_capability_test_records(self, capability_name: CapabilityName) -> CapabilityTestRecords:
        if capability_name in self.capability_test_records:
            return self.capability_test_records[capability_name]
        new_ctr = CapabilityTestRecords(capability_name)
        self.capability_test_records[capability_name] = new_ctr
        return new_ctr

    def find_associated_capabilities(self, test_record: TestRecord) -> Iterable[CapabilityName]:
        associated_capabilities: List[CapabilityName] = list()
        primary_capability_name = test_record.test_id[:1]  # Just a stop gap in place of looking these up from a database
        if not self.grouping_by_upgrade:
            if test_record.upgrade == 'upgrade-minor':
                associated_capabilities.append(f'y-upgrade: {primary_capability_name}')
            elif test_record.upgrade == 'upgrade-micro':
                associated_capabilities.append(f'z-upgrade: {primary_capability_name}')
            else:
                associated_capabilities.append(f'install: {primary_capability_name}')
        else:
            associated_capabilities.append(primary_capability_name)

        return associated_capabilities

    def add_test_record(self, test_record: TestRecord):
        # Find all capabilities with which the test should be associated.
        for capability_name in self.find_associated_capabilities(test_record):
            self.get_capability_test_records(capability_name).add_test_record(test_record)

    def assessment(self) -> TestRecordAssessment:
        return TestRecord.aggregate_assessment([ctr.assessment() for ctr in self.capability_test_records.values()])


class EnvironmentTestRecords:

    COMPONENT_NAME_PATTERN = re.compile(r'\[[^\\]+?\]')

    def __init__(self, name: EnvironmentName, grouping_by_upgrade=False, platform: str = '', arch: str = '', network: str = '', upgrade: str = ''):
        self.all_test_record_uuids: Set[TestUUID] = set()
        self.name: EnvironmentName = name
        self.grouping_by_upgrade = grouping_by_upgrade
        self.component_test_records: Dict[ComponentName, ComponentTestRecords] = dict()
        self.platform = platform
        self.arch = arch
        self.network = network
        self.upgrade = upgrade

        # Within a given environment, results for a specific TestId can be added several times.
        # For example, if the environment is grouping by Platform+Arch+Network, the
        # environment model add to this EnvironmentTestRecords multiple identical TestId values
        # but each will be associated with a different upgrade.
        # So EnvironmentTestRecords work based on TestUUID which is a combination of
        # NURP+TestID which makes it completely unique for the Environment (i.e. the query
        # to the database must be grouping on these attributes to ensure that only a single
        # such row exists in the results).
        # All regression is assessed on TestUUID comparisons between basis and sample.
        self.all_test_records: Dict[TestUUID, TestRecord] = dict()

    def get_breadcrumb(self) -> str:
        return f"[cloud='{self.platform}' arch='{self.arch}' network='{self.network}' upgrade='{self.upgrade}']"

    def get_component_test_records(self, component_name: ComponentName) -> ComponentTestRecords:
        if component_name in self.component_test_records:
            return self.component_test_records[component_name]
        new_ctr = ComponentTestRecords(component_name, self.grouping_by_upgrade)
        self.component_test_records[component_name] = new_ctr
        return new_ctr

    def find_associated_components(self, test_record: TestRecord) -> Iterable[ComponentName]:
        return re.findall(EnvironmentTestRecords.COMPONENT_NAME_PATTERN, test_record.test_name)

    def add_test_record(self, test_record: TestRecord) -> Iterable[ComponentName]:
        self.all_test_records[test_record.test_uuid] = test_record
        self.all_test_record_uuids.add(test_record.test_uuid)
        # Find a list of all component names to which this test belongs.
        # Register the test with each component name.
        add_to_components = self.find_associated_components(test_record)
        for component_name in add_to_components:
            component_test_records = self.get_component_test_records(component_name)
            component_test_records.add_test_record(test_record)
        return add_to_components

    def get_component_names(self):
        return self.component_test_records.keys()

    def assessment(self) -> TestRecordAssessment:
        return TestRecord.aggregate_assessment([ctr.assessment() for ctr in self.component_test_records.values()])

    def build_mass_assessment_cache(self, basis_environment_test_records: "EnvironmentTestRecords", alpha: float, regression_when_missing: bool):

        basis_test_uuids = set(basis_environment_test_records.all_test_record_uuids)
        sample_test_uuids = set(self.all_test_record_uuids)
        test_uuids_not_in_sample = basis_test_uuids.difference(sample_test_uuids)

        for test_uuid in test_uuids_not_in_sample:
            basis_test_record = basis_environment_test_records.all_test_records[test_uuid]
            # TODO: if a test_id has been officially deprecated by staff, do not add this record
            place_holder_record = TestRecord(
                platform=basis_test_record.platform,
                network=basis_test_record.network,
                upgrade=basis_test_record.upgrade,
                arch=basis_test_record.arch,
                assessment=TestRecordAssessment.MISSING_IN_SAMPLE if regression_when_missing else TestRecordAssessment.NOT_SIGNIFICANT,
                test_id=basis_test_record.test_id,
                test_name=basis_test_record.test_name,
            )
            # Adding these missing tests ensures that the sample environment has a superset
            # of components and capabilities compared to the basis environment.
            self.add_test_record(place_holder_record)

        for test_uuid, sample_test_record in self.all_test_records.items():
            if test_uuid in basis_environment_test_records.all_test_records:
                basis_test_record = basis_environment_test_records.all_test_records[test_uuid]
                sample_test_record.compute_assessment(basis_test_record, alpha, regression_when_missing)
            else:
                sample_test_record.cached_assessment = TestRecordAssessment.MISSING_IN_BASIS


class EnvironmentModel:

    def get_ordered_environment_names(self):
        return sorted(list(self.environment_test_records.keys()))

    def get_ordered_component_names(self):
        return sorted(self.all_component_names)

    def get_environment_test_records(self, env_name, grouping_by_upgrades=False, **kwargs) -> EnvironmentTestRecords:
        if env_name in self.environment_test_records:
            return self.environment_test_records[env_name]
        new_etr = EnvironmentTestRecords(env_name, grouping_by_upgrades, **kwargs)
        self.environment_test_records[env_name] = new_etr
        return new_etr

    def __init__(self):
        self.environment_test_records: Dict[EnvironmentName, EnvironmentTestRecords] = dict()
        self.all_component_names: Set[ComponentName] = set()

    def read_in_query(self, query, group_by: str):
        group_by_elements = group_by.split(',')
        group_by_network = 'network' in group_by_elements
        group_by_upgrade = 'upgrade' in group_by_elements
        group_by_arch = 'arch' in group_by_elements
        group_by_platform = 'cloud' in group_by_elements or 'platform' in group_by_elements

        if not any((group_by_network, group_by_upgrade, group_by_arch, group_by_platform)):
            raise IOError('Invalid group-by values')

        for row in query.execute():

            env_attributes: Dict[str, str] = dict()
            env_name_components: List[str] = list()  # order of this list reflects in column heading names
            if group_by_platform:
                env_attributes['platform'] = row.platform
                env_name_components.append(row.platform)
            if group_by_arch:
                env_attributes['arch'] = row.arch
                env_name_components.append(row.arch)
            if group_by_network:
                env_attributes['network'] = row.network
                env_name_components.append(row.network)
            if group_by_upgrade:
                env_attributes['upgrade'] = row.upgrade
                env_name_components.append(row.upgrade)

            env_name = ' '.join(env_name_components)
            flake_count = row.flake_count
            success_count = row.success_count
            # In rare circumstances, based on the date range selected, it is possible for a failed test run to not be included
            # in the query while the success run (including a flake_count=1 reflecting the preceding, but un-selected
            # failure) is included. This could make total_count - flake_count a negative value.
            total_count_minus_flakes = max(success_count, row.total_count - flake_count)

            r = TestRecord(
                platform=row.platform,
                network=row.network,
                upgrade=row.upgrade,
                arch=row.arch,
                test_id=row.test_id,
                test_name=row.test_name,
                success_count=success_count,
                failure_count=total_count_minus_flakes - success_count,
                total_count_minus_flakes=total_count_minus_flakes,
                flake_count=flake_count,
            )

            environment_test_records = self.get_environment_test_records(env_name, group_by_upgrade, **env_attributes)
            component_name_modified = environment_test_records.add_test_record(r)
            self.all_component_names.update(component_name_modified)

    def build_mass_assessment_cache(self, basis_model: "EnvironmentModel", alpha: float = 0.05, regression_when_missing: bool=True):

        # Get a list of all environments - including both basis and sample in case
        # there is one in basis that no longer exists in samples.
        all_names = set(self.get_ordered_environment_names())
        all_names.update(basis_model.get_ordered_environment_names())

        for environment_name in all_names:
            sample_environment_test_records = self.get_environment_test_records(environment_name)
            basis_environment_test_records = basis_model.get_environment_test_records(environment_name)
            sample_environment_test_records.build_mass_assessment_cache(basis_environment_test_records, alpha, regression_when_missing)
