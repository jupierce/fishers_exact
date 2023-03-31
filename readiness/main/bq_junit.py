import time

from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import expression

from google.cloud import bigquery, bigquery_storage

from typing import Dict, Optional, Iterable, List, Set, NamedTuple
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
    flat_variants = __table__.c.flat_variants
    testsuite = __table__.c.testsuite

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
concat = func.concat
array_to_string = func.array_to_string
sum = func.sum

EnvironmentName = str
ComponentName = str
CapabilityName = str

# When a typical user should have the concept of a test being the "same test", the
# test ids for individual test runs should match. Initially, this was identical to the
# test's name, but complexities were discovered:
# - If a test with the same name is run in two different suites, it should result in different test ids.
# - If a test runs on
TestId = str


TestUUID = str
TestName = str


class TestRecordAssessment(Enum):
    EXTREME_REGRESSION = (-3, 'Regression with >15% pass rate change', 'fire.png')
    SIGNIFICANT_REGRESSION = (-2, 'Significant regression', 'red.png')
    MISSING_IN_SAMPLE = (-1, 'No test runs in sample', 'red-question-mark.png')
    NOT_SIGNIFICANT = (0, 'No significant deviation', 'green.png')
    MISSING_IN_BASIS = (1, 'No records in basis data', 'green.png')
    SIGNIFICANT_IMPROVEMENT = (2, 'Significant improvement', 'darkgreen-heart.png')

    def __init__(self, val: int, description: str, image_path: str):
        self.val = val
        self.description = description
        self.image_path = image_path


class TestRecord:

    @classmethod
    def aggregate_assessment(cls, test_assessments: Iterable[TestRecordAssessment]):
        overall_assessment: TestRecordAssessment = TestRecordAssessment.NOT_SIGNIFICANT
        found_missing_in_sample = False
        sorted_test_assessments = sorted(list(test_assessments), key=lambda x: x.val)  # Sort from low score to high
        for test_assessment in sorted_test_assessments:
            # A single significant regression in any test means an overall regression
            # This is why the list is sorted ahead of time. Return the worst signal of the group.
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
                 env_name: str,
                 platform: str,
                 network: str,
                 arch: str,
                 upgrade: str,
                 flat_variants: str,
                 test_id: TestId,
                 testsuite: str,
                 test_name: TestName,
                 success_count: int = 0,
                 failure_count: int = 0,
                 total_count_minus_flakes: int = 0,
                 flake_count: int = 0,
                 assessment: TestRecordAssessment = None):
        self.env_name = env_name
        self.test_id = test_id
        self.testsuite = testsuite
        self.test_name = test_name
        self.success_count = success_count
        self.failure_count = failure_count
        self.total_count_minus_flakes = total_count_minus_flakes
        self.flake_count = flake_count
        self.cached_assessment = assessment
        self.platform = platform
        self.network = network
        self.arch = arch
        self.upgrade = upgrade
        self.flat_variants = flat_variants
        self.pass_percentage = 0.0 if self.total_count_minus_flakes == 0 else 100 * self.success_count / self.total_count_minus_flakes
        self.test_uuid: TestUUID = ':'.join((platform, network, arch, upgrade, flat_variants, test_id))

    def assessment(self) -> TestRecordAssessment:
        if self.cached_assessment is None:
            return TestRecordAssessment.MISSING_IN_BASIS
        return self.cached_assessment

    @property
    def success_rate_str(self) -> str:
        return '{:.2f}'.format(self.pass_percentage)

    def compute_assessment(self, basis_test_record: "TestRecord", alpha: float, regression_when_missing: bool, pity_factor: float = 0.05):
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

            basis_pass_percentage = basis_test_record.pass_percentage
            sample_pass_percentage = self.pass_percentage
            improved = sample_pass_percentage >= basis_pass_percentage

            if improved:
                significant = fisher_significant(
                    basis_test_record.failure_count,
                    basis_test_record.success_count,
                    self.failure_count,
                    self.success_count,
                    alpha=alpha,
                )
            else:
                if basis_pass_percentage - sample_pass_percentage < pity_factor:
                    # Until the difference in pass rate has decreased beyond the pity
                    # factory, allow the square to remain green.
                    significant = False
                else:
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

    __slots__ = [
        'test_id',
        'test_records',
        'canonical_test_name'
    ]

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
        self.test_records.setdefault(test_record.test_uuid, test_record)
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
        trs = self.test_record_sets.get(test_id, None)
        if not trs:
            trs = TestRecordSet(test_id)
            self.test_record_sets[test_id] = trs
        return trs

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
        ctr = self.capability_test_records.get(capability_name, None)
        if not ctr:
            ctr = CapabilityTestRecords(capability_name)
            self.capability_test_records[capability_name] = ctr
        return ctr

    def find_associated_capabilities(self, test_record: TestRecord) -> Iterable[CapabilityName]:
        associated_capabilities: List[CapabilityName] = list()

        primary_capability_name = test_record.test_id[-1:]  # Just a stop gap in place of looking these up from a database

        if test_record.testsuite == 'openshift-tests-upgrade':
            # Deads requests that this testsuite be broken out into its own
            # capability.
            associated_capabilities.append(f'openshift-tests-upgrade: {primary_capability_name}')
        elif not self.grouping_by_upgrade:
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

    def __init__(self, reference: TestRecord, grouping_by_upgrade: bool, platform: str = '', arch: str = '', network: str = '', upgrade: str = '', variant: str = ''):
        self.all_test_record_uuids: Set[TestUUID] = set()
        self.name: EnvironmentName = reference.env_name
        self.reference = reference
        self.grouping_by_upgrade = grouping_by_upgrade
        self.component_test_records: Dict[ComponentName, ComponentTestRecords] = dict()
        self.platform = platform
        self.arch = arch
        self.network = network
        self.upgrade = upgrade
        self.variant = variant

        # Within a given environment, results for a specific TestId can be added several times.
        # For example, if the environment is grouping by Platform+Arch+Network, the
        # environment model add to this EnvironmentTestRecords multiple identical TestId values
        # but each will be associated with a different upgrade.
        # So EnvironmentTestRecords work based on TestUUID which is a combination of
        # NURPP+TestID which makes it completely unique for the Environment (i.e. the query
        # to the database must be grouping on these attributes to ensure that only a single
        # such row exists in the results).
        # All regression is assessed on TestUUID comparisons between basis and sample.
        self.all_test_records: Dict[TestUUID, TestRecord] = dict()

    def get_breadcrumb(self) -> str:
        return f"[cloud='{self.platform}' arch='{self.arch}' network='{self.network}' upgrade='{self.upgrade}' variant='{self.variant}']"

    def get_component_test_records(self, component_name: ComponentName) -> ComponentTestRecords:
        ctr = self.component_test_records.get(component_name, None)
        if not ctr:
            ctr = ComponentTestRecords(component_name, self.grouping_by_upgrade)
            self.component_test_records[component_name] = ctr
        return ctr

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

    def build_mass_assessment_cache(self, basis_environment_test_records: "EnvironmentTestRecords", alpha: float, regression_when_missing: bool, pity_factor: float = 0.05):

        basis_test_uuids = set(basis_environment_test_records.all_test_record_uuids)
        sample_test_uuids = set(self.all_test_record_uuids)
        test_uuids_not_in_sample = basis_test_uuids.difference(sample_test_uuids)

        for test_uuid in test_uuids_not_in_sample:
            basis_test_record = basis_environment_test_records.all_test_records[test_uuid]
            # TODO: if a test_id has been officially deprecated by staff, do not add this record
            place_holder_record = TestRecord(
                env_name=basis_test_record.env_name,
                platform=basis_test_record.platform,
                network=basis_test_record.network,
                upgrade=basis_test_record.upgrade,
                arch=basis_test_record.arch,
                flat_variants=basis_test_record.flat_variants,
                assessment=TestRecordAssessment.MISSING_IN_SAMPLE if regression_when_missing else TestRecordAssessment.NOT_SIGNIFICANT,
                test_id=basis_test_record.test_id,
                test_name=basis_test_record.test_name,
                testsuite=basis_test_record.testsuite,
            )
            # Adding these missing tests ensures that the sample environment has a superset
            # of components and capabilities compared to the basis environment.
            self.add_test_record(place_holder_record)

        for test_uuid, sample_test_record in self.all_test_records.items():
            basis_test_record = basis_environment_test_records.all_test_records.get(test_uuid, None)
            if basis_test_record:
                sample_test_record.compute_assessment(basis_test_record, alpha=alpha, regression_when_missing=regression_when_missing, pity_factor=pity_factor)
            else:
                sample_test_record.cached_assessment = TestRecordAssessment.MISSING_IN_BASIS


class ModelColumn(NamedTuple):
    name: str
    offset: int  # column offset when looking at dataframe


# Column offsets must exactly match the order get_environment_query_scan
COLUMN_NETWORK = ModelColumn('network', 0)
COLUMN_UPGRADE = ModelColumn('upgrade', 1)
COLUMN_ARCH = ModelColumn('arch', 2)
COLUMN_PLATFORM = ModelColumn('platform', 3)
COLUMN_TEST_ID = ModelColumn('test_id', 4)
COLUMN_FLAT_VARIANTS = ModelColumn('flat_variants', 5)
COLUMN_TESTSUITE = ModelColumn('testsuite', 6)
COLUMN_TOTAL_COUNT = ModelColumn('total_count', 7)
COLUMN_TEST_NAME = ModelColumn('test_name', 8)
COLUMN_SUCCESS_COUNT = ModelColumn('success_count', 9)
COLUMN_FLAKE_COUNT = ModelColumn('flake_count', 10)
COLUMN_ENV_NAME = ModelColumn('env_name', 11)
LAST_REAL_COLUMN = COLUMN_ENV_NAME
COLUMN_COMPUTED_FAILURE_COUNT_OPTION = ModelColumn('failure_count_option', LAST_REAL_COLUMN.offset+1)
COLUMN_COMPUTED_TOTAL_MINUS_FLAKES_OPTION = ModelColumn('total_minus_flakes_option', LAST_REAL_COLUMN.offset+2)


class EnvironmentModel:

    def get_ordered_environment_names(self):
        return sorted(list(self.environment_test_records.keys()))

    def get_ordered_component_names(self):
        return sorted(self.all_component_names)

    def get_environment_test_records(self, env_name: str, reference: TestRecord = None) -> EnvironmentTestRecords:
        etr = self.environment_test_records.get(env_name, None)
        if not etr:
            etr = EnvironmentTestRecords(reference, self.group_by_upgrade,
                                         platform=reference.platform if self.group_by_platform else '',
                                         arch=reference.arch if self.group_by_arch else '',
                                         network=reference.network if self.group_by_network else '',
                                         upgrade=reference.upgrade if self.group_by_upgrade else '',
                                         variant=reference.variant if self.group_by_variant else '',
                                         )
            self.environment_test_records[env_name] = etr
        return etr

    def __init__(self, name, group_by):
        self.environment_test_records: Dict[EnvironmentName, EnvironmentTestRecords] = dict()
        self.all_component_names: Set[ComponentName] = set()
        self.model_name = name
        self.group_by = group_by

        group_by_elements = self.group_by.split(',')
        self.group_by_network = 'network' in group_by_elements
        self.group_by_upgrade = 'upgrade' in group_by_elements
        self.group_by_arch = 'arch' in group_by_elements
        self.group_by_variant = 'variant' in group_by_elements
        self.group_by_platform = 'cloud' in group_by_elements or 'platform' in group_by_elements

        if not any((self.group_by_network, self.group_by_upgrade, self.group_by_arch, self.group_by_platform, self.group_by_variant)):
            raise IOError('Invalid group-by values')

    def get_environment_query_scan(self):
        j = Junit
        env_name_components = []

        def append_env_component(col):
            if env_name_components:
                env_name_components.append(' ')
            env_name_components.append(col)

        if self.group_by_platform:
            append_env_component(j.platform)
        if self.group_by_arch:
            append_env_component(j.arch)
        if self.group_by_network:
            append_env_component(j.network)
        if self.group_by_upgrade:
            append_env_component(j.upgrade)
        if self.group_by_variant:
            append_env_component(j.offset)

        base_select = select(
            j.network,
            j.upgrade,
            j.arch,
            j.platform,
            j.test_id,
            j.flat_variants,
            any_value(j.testsuite).label(COLUMN_TESTSUITE.name),
            count(j.test_id).label(COLUMN_TOTAL_COUNT.name),
            concat(any_value(j.testsuite), ":", any_value(j.test_name)).label(COLUMN_TEST_NAME.name),
            sum(j.success_val).label(COLUMN_SUCCESS_COUNT.name),
            sum(j.flake_count).label(COLUMN_FLAKE_COUNT.name),
            concat(*env_name_components).label(COLUMN_ENV_NAME.name)
        )

        return base_select.group_by(
            j.test_id,
            j.platform,
            j.network,
            j.upgrade,
            j.arch,
            j.flat_variants,
        ).where(
            or_(
                j.testsuite == 'openshift-tests',
                j.testsuite == 'openshift-tests-upgrade'
            )
        )

    def read_in_query(self, query):
        bq_client = bigquery.Client(project='openshift-gce-devel')
        raw_query_string = str(query.compile(_junit_table_engine, compile_kwargs={"literal_binds": True}))

        start_overhead = time.time()
        #df = bq_client.query(raw_query_string).to_dataframe(create_bqstorage_client=False, progress_bar_type='tqdm')
        df = bq_client.query(raw_query_string).to_dataframe(create_bqstorage_client=True, progress_bar_type='tqdm')
        # df = pandas.read_gbq(raw_query_string, dialect='standard', use_bqstorage_api=True)
        # print(df.head())
        df = df.assign(failure_count_option=df['total_count'] - df['flake_count'] - df['success_count'])
        df = df.assign(total_minus_flakes_option=df['total_count'] - df['flake_count'])

        test_records: List[TestRecord] = [
            TestRecord(
                env_name=dfr[COLUMN_ENV_NAME.offset],
                platform=dfr[COLUMN_PLATFORM.offset],
                network=dfr[COLUMN_NETWORK.offset],
                upgrade=dfr[COLUMN_UPGRADE.offset],
                flat_variants=dfr[COLUMN_FLAT_VARIANTS.offset],
                arch=dfr[COLUMN_ARCH.offset],
                test_id=dfr[COLUMN_TEST_ID.offset],
                testsuite=dfr[COLUMN_TESTSUITE.offset],
                test_name=dfr[COLUMN_TEST_NAME.offset],
                success_count=dfr[COLUMN_SUCCESS_COUNT.offset],
                failure_count=max(0, dfr[COLUMN_COMPUTED_FAILURE_COUNT_OPTION.offset]),
                total_count_minus_flakes=max(dfr[COLUMN_SUCCESS_COUNT.offset], dfr[COLUMN_COMPUTED_TOTAL_MINUS_FLAKES_OPTION.offset]),
                flake_count=dfr[COLUMN_FLAKE_COUNT.offset],
            )
            for dfr in zip(
                df[COLUMN_NETWORK.name],
                df[COLUMN_UPGRADE.name],
                df[COLUMN_ARCH.name],
                df[COLUMN_PLATFORM.name],
                df[COLUMN_TEST_ID.name],
                df[COLUMN_FLAT_VARIANTS.name],
                df[COLUMN_TESTSUITE.name],
                df[COLUMN_TOTAL_COUNT.name],
                df[COLUMN_TEST_NAME.name],
                df[COLUMN_SUCCESS_COUNT.name],
                df[COLUMN_FLAKE_COUNT.name],
                df[COLUMN_ENV_NAME.name],
                df[COLUMN_COMPUTED_FAILURE_COUNT_OPTION.name],
                df[COLUMN_COMPUTED_TOTAL_MINUS_FLAKES_OPTION.name],
            )
        ]

        total_time = 0.0
        for r in test_records:
            start = time.time()
            environment_test_records = self.get_environment_test_records(r.env_name, reference=r)
            component_name_modified = environment_test_records.add_test_record(r)
            self.all_component_names.update(component_name_modified)
            total_time += time.time() - start
        print(f'Tree building time: {total_time}')
        print(f'Overhead time: {time.time() - start_overhead - total_time}')

    def build_mass_assessment_cache(self, basis_model: "EnvironmentModel", alpha: float = 0.05, regression_when_missing: bool = True, pity_factor: float = 0.05):
        # Get a list of all environments - including both basis and sample in case
        # there is one in basis that no longer exists in samples.
        all_names = set(self.get_ordered_environment_names())
        all_names.update(basis_model.get_ordered_environment_names())

        # Make sure all basis environments / components / capabilities exist in the sample environment
        for environment_name, env in basis_model.environment_test_records.items():
            self.get_environment_test_records(environment_name, env.reference)

        for environment_name, env in self.environment_test_records.items():
            basis_model.get_environment_test_records(environment_name, env.reference)   # Make sure the sample environments / components / capabilities exist in basis
            env.build_mass_assessment_cache(basis_model.environment_test_records[environment_name], alpha=alpha, regression_when_missing=regression_when_missing, pity_factor=pity_factor)
