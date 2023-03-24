import re
import math
import mmap
import traceback
from concurrent.futures import ThreadPoolExecutor

from typing import Dict, NamedTuple, List, Tuple, Any, Optional, Iterable
from enum import Enum

from django.utils.html import format_html
from django.http import HttpResponse
import django_tables2 as tables

from .bq_junit import Junit, select, sum, count, any_value

import fast_fisher.fast_fisher_cython

from django.shortcuts import render
from readiness.settings import BASE_DIR

from google.cloud import bigquery


EnvironmentKey = str
TestId = str
ComponentName = str
CapabilityName = str

MAX_CELL_SAMPLES = 512   # must be a power of 2 and divisible by 8
MULTIPLICATION_SHIFTS = int(math.log2(MAX_CELL_SAMPLES))

if (1 << MULTIPLICATION_SHIFTS) != MAX_CELL_SAMPLES or MAX_CELL_SAMPLES//8*8 != MAX_CELL_SAMPLES:
    print(f'MAX_CELL_SAMPLES must be a power of 2 and divisible by 8')
    exit(1)

ALPHA = 0.05

try:
    table_bin = open(f'{BASE_DIR}/../table.bin', 'rb')
    table_mmap = mmap.mmap(table_bin.fileno(), length=0, access=mmap.ACCESS_READ)
except:
    print(f'WARNING: Precomputed fishers will not be used.')
    table_mmap = None
    traceback.print_exc()


def fisher_offset(a, b, c, d) -> int:
    # TODO: use chi^2 instead of normalizing to 500 samples?
    if a > 500 or b > 500:
        reducer = min((500 / max(a, 1)), (500 / max(b, 1)))  # use max in case one of the values is 0
        a = int(a * reducer)
        b = int(b * reducer)
    if c > 500 or d > 500:
        reducer = min((500 / max(c, 1)), (500 / max(d, 1)))
        c = int(c * reducer)
        d = int(d * reducer)
    return (a << (MULTIPLICATION_SHIFTS * 3)) + (b << (MULTIPLICATION_SHIFTS * 2)) + (c << (MULTIPLICATION_SHIFTS * 1)) + d


def fisher_significant(a, b, c, d) -> bool:
    if (not table_mmap) or a > 500 or b > 500 or c > 500 or d > 500:
        return fast_fisher.fast_fisher_cython.fisher_exact(a, b, c, d, alternative='greater') < 0.05
    # Otherwise, look it up faster in the precomputed data
    offset = fisher_offset(a, b, c, d)
    bin_offset = offset // 8
    bit_shift = 7 - (offset % 8)
    return table_mmap[bin_offset] & (1 << bit_shift) > 0


class TestRecord(NamedTuple):
    test_id: Optional[TestId]
    test_name: Optional[str]
    success_count: int
    failure_count: int
    total_count: int
    flake_count: int


NO_TESTS = TestRecord(
    test_id=None,
    test_name=None,
    success_count=0,
    failure_count=0,
    total_count=0,
    flake_count=0,
)


class Conclusion(Enum):
    MISSING_IN_BASIS = 1
    SIGNIFICANT_IMPROVEMENT = 2
    SIGNIFICANT_REGRESSION = 3


ConclusionMap = Dict[EnvironmentKey, Dict[TestId, Conclusion]]


def has_regression(records: Iterable[TestRecord], conclusions: Dict[TestId, Conclusion]) -> bool:
    """
    :param records: The records to analyze
    :param conclusions: The conclusions for all tests analyzed
    :return: Returns True if even on TestRecord is a regression.
    """
    for record in records:
        conclusion = conclusions.get(record.test_id, None)
        if conclusion == Conclusion.SIGNIFICANT_REGRESSION:
            return True
    return False


class CapabilityRecords(NamedTuple):
    name: CapabilityName
    test_records: Dict[TestId, TestRecord]

    def has_regressed(self, conclusions: Dict[TestId, Conclusion]) -> bool:
        return has_regression(self.test_records.values(), conclusions=conclusions)

    def register_test_record(self, test_record: TestRecord):
        self.test_records[test_record.test_id] = test_record


ByCapability = Dict[CapabilityName, CapabilityRecords]


class ComponentRecords(NamedTuple):
    name: ComponentName
    test_records: Dict[TestId, TestRecord]
    capabilities: ByCapability

    def has_regressed(self, conclusions: Dict[TestId, Conclusion]) -> bool:
        return has_regression(self.test_records.values(), conclusions=conclusions)

    def register_test_record(self, test_record: TestRecord):
        self.test_records[test_record.test_id] = test_record

    def register_test_record_capability(self, capability_name: CapabilityName, test_record: TestRecord):
        if capability_name not in self.capabilities:
            self.capabilities[capability_name] = CapabilityRecords(capability_name, dict())
        self.capabilities[capability_name].test_records[test_record.test_id] = test_record


ById = Dict[TestId, TestRecord]
ByComponent = Dict[ComponentName, ComponentRecords]


def index(request):
    return render(request, "main/index.html")


component_pattern = re.compile(r'\[[^\\]+?\]')


def categorize(sql_query) -> Dict[EnvironmentKey, Tuple[ById, ByComponent]]:
    environments: Dict[EnvironmentKey, Tuple[ById, ByComponent]] = dict()

    for row in sql_query.execute():
        environment_name = ' '.join((row.network, row.upgrade, row.arch, row.platform))

        if environment_name not in environments:
            by_id: ById = dict()
            by_component: ByComponent = dict()
            environments[environment_name] = (by_id, by_component)
        else:
            by_id, by_component = environments[environment_name]

        if row.success_count is None:
            # Not sure why, but bigquery can return None for some SUMs.
            # Potentially because success_val was not set on all original rows
            continue

        flake_count = row.flake_count
        success_count = row.success_count
        # In rare circumstances, based on the date range selected, it is possible for a failed test run to not be included
        # in the query while the success run (including a flake_count=1 reflecting the preceding, but un-selected
        # failure) is included. This could make total_count - flake_count a negative value.
        total_count = max(success_count, row.total_count - flake_count)

        r = TestRecord(
            test_id=row.test_id,
            test_name=row.test_name,
            success_count=success_count,
            failure_count=total_count - success_count,
            total_count=total_count,
            flake_count=flake_count
        )
        by_id[r.test_id] = r
        labels = re.findall(component_pattern, r.test_name)
        for label in labels:
            if label not in by_component:
                by_component[label] = ComponentRecords(name=label, test_records=dict(), capabilities=dict())
            by_component[label].register_test_record(r)

            capability = r.test_id[:1]
            by_component[label].register_test_record_capability(capability, r)

    return environments


def calculate_conclusions(basis_envs: Dict[EnvironmentKey, Tuple[ById, ByComponent]], sample_envs: Dict[EnvironmentKey, Tuple[ById, ByComponent]]) -> ConclusionMap:
    conclusions_map: ConclusionMap = dict()
    for nurp_name in sample_envs:
        basis_by_id, _ = basis_envs.get(nurp_name, (dict(), dict()))
        samples_by_id, _ = sample_envs[nurp_name]
        by_conclusion: Dict[TestId, Conclusion] = dict()
        conclusions_map[nurp_name] = by_conclusion

        for test_id, sample_result in samples_by_id.items():
            basis_result = basis_by_id.get(test_id, None)
            if not basis_result or basis_result.total_count == 0:
                by_conclusion[test_id] = Conclusion.MISSING_IN_BASIS
                continue

            basis_pass_percentage = basis_result.success_count / basis_result.total_count
            sample_pass_percentage = sample_result.success_count / sample_result.total_count
            improved = sample_pass_percentage >= basis_pass_percentage

            significant = fisher_significant(
                sample_result.failure_count,
                sample_result.success_count,
                basis_result.failure_count,
                basis_result.success_count,
            )

            if significant:
                by_conclusion[test_id] = Conclusion.SIGNIFICANT_IMPROVEMENT if improved else Conclusion.SIGNIFICANT_REGRESSION

    return conclusions_map


class ImageColumnLink(NamedTuple):
    image_path: str
    height: Optional[int] = None
    width: Optional[int] = None
    href: Optional[str] = None
    href_params: Optional[Dict[str, str]] = None


class ImageColumn(tables.Column):

    def render(self, value: ImageColumnLink):
        image_path = value.image_path

        height_attr = ''
        if value.height:
            height_attr = f'height="{value.height}" '

        width_attr = ''
        if value.width:
            width_attr = f'width="{value.height}" '

        content = f'<img {height_attr}{width_attr} src="/static/{image_path}"></img>'
        if value.href:
            content = f'<a href="{value.href}?{dict_to_params_url(value.href_params)}">{content}</a>'
        return format_html(content)


def dict_to_params_url(params):
    if not params:
        return ''
    return '&'.join([f'{key}={value}' for key, value in params.items()])


class ProwjobTable(tables.Table):
    prowjob_name = tables.Column()
    basis_info = tables.Column()
    sample_info = tables.Column()
    individual_job_regression = ImageColumn()

    class Meta:
        attrs = {"class": "paleblue"}


class AllComponentsTable(tables.Table):
    name = tables.Column()

    def __init__(self, data, extra_columns, new_key=None, params=None):
        super().__init__(data, extra_columns=extra_columns)
        self.params = params
        self.new_key = new_key

    class Meta:
        attrs = {"class": "paleblue"}

    def render_name(self, value):
        if self.params is None:
            params = dict()
        else:
            params = dict(self.params)
        if self.new_key:
            params[self.new_key] = value

        if params:
            params_str = dict_to_params_url(params)
            return format_html(f'<a href="/main/report?{params_str}">{value}</a>')
        else:
            return value


COLUMN_TEST_NAME = 'test_name'
COLUMN_TOTAL_COUNT = 'total_count'
COLUMN_SUCCESS_COUNT = 'success_count'
COLUMN_FLAKE_COUNT = 'flake_count'


def report(request):

    def insufficient_sanitization(parameter_name: str, parameter_default: Optional[str]=None) -> Optional[str]:
        val = request.GET.get(parameter_name, parameter_default)
        if val is not None and ("'" in val or '"' in val or '\\' in val):
            raise IOError('Sanitization failure')
        return val

    basis_release = insufficient_sanitization('basis_release')
    basis_start_dt = insufficient_sanitization('basis_start_dt')
    basis_end_dt = insufficient_sanitization('basis_end_dt')

    sample_release = insufficient_sanitization('sample_release')
    sample_start_dt = insufficient_sanitization('sample_start_dt')
    sample_end_dt = insufficient_sanitization('sample_end_dt')

    target_component_name = insufficient_sanitization('component', None)
    target_capability_name = insufficient_sanitization('capability', None)
    target_test_id = insufficient_sanitization('test_id', None)
    target_platform_name = insufficient_sanitization('platform', None)
    target_upgrade_name = insufficient_sanitization('upgrade', None)
    target_arch_name = insufficient_sanitization('arch', None)
    target_network_name = insufficient_sanitization('network', None)

    j = Junit
    pqb = select(
        j.network,
        j.upgrade,
        j.arch,
        j.platform,
        j.test_id,
        any_value(j.test_name).label(COLUMN_TEST_NAME),
        count(j.test_id).label(COLUMN_TOTAL_COUNT),
        sum(j.success_val).label(COLUMN_SUCCESS_COUNT),
        sum(j.flake_count).label(COLUMN_FLAKE_COUNT),
    ).group_by(
        j.network,
        j.upgrade,
        j.arch,
        j.platform,
        j.test_id,
    )

    def assert_all_set(lt: Iterable, error_msg: str):
        if not all(lt):
            raise ValueError(error_msg)

    assert_all_set((basis_start_dt, basis_end_dt, basis_release), 'At least one basis coordinate has not been specified')

    assert_all_set((sample_start_dt, sample_end_dt, sample_release), 'At least one sample coordinate has not been specified')

    if any((target_network_name, target_upgrade_name, target_arch_name, target_platform_name, target_test_id)):
        assert_all_set((target_network_name, target_upgrade_name, target_arch_name, target_platform_name), 'Elements of primary drill key were not specified')

        pqb = pqb.filter(
            j.network == target_network_name,
            j.upgrade == target_upgrade_name,
            j.arch == target_arch_name,
            j.platform == target_platform_name,
        )

        if target_test_id:
            pqb = pqb.filter(
                j.test_id == target_test_id
            )

    # q = f'''
    #     SELECT CONCAT(network, " ", upgrade, " ", arch, " ", platform) as nurp, test_id, ANY_VALUE(test_name) as test_name, COUNT(*) as total_count, SUM(success_val) as success_count, SUM(flake_count) as flake_count
    #     FROM `openshift-gce-devel.ci_analysis_us.junit`
    #     WHERE modified_time >= DATETIME(TIMESTAMP "{basis_start_dt}:00+00") AND modified_time < DATETIME(TIMESTAMP "{basis_end_dt}:00+00") {target_nurp_filter} {target_test_filter}
    #           AND branch = "{basis_release}" AND file_path NOT LIKE "%/junit_operator.xml" GROUP BY network, upgrade, arch, platform, test_id
    # '''
    basis_query = pqb.filter(
        j.modified_time >= j.format_modified_time(basis_start_dt),
        j.modified_time < j.format_modified_time(basis_end_dt),
        j.branch == basis_release
    )

    # q = f'''
    #     SELECT CONCAT(network, " ", upgrade, " ", arch, " ", platform) as nurp, test_id, ANY_VALUE(test_name) as test_name, COUNT(*) as total_count, SUM(success_val) as success_count, SUM(flake_count) as flake_count
    #     FROM `openshift-gce-devel.ci_analysis_us.junit`
    #     WHERE modified_time >= DATETIME(TIMESTAMP "{sample_start_dt}:00+00") AND modified_time < DATETIME(TIMESTAMP "{sample_end_dt}:00+00") {target_nurp_filter} {target_test_filter}
    #           AND branch = "{sample_release}" AND file_path NOT LIKE "%/junit_operator.xml" GROUP BY network, upgrade, arch, platform, test_id
    # '''

    sample_query = pqb.filter(
        j.modified_time >= j.format_modified_time(sample_start_dt),
        j.modified_time < j.format_modified_time(sample_end_dt),
        j.branch == sample_release
    )

    # Performance note: the .query() calls return relatively quickly. Bigquery is doing the calculations
    # and python will only block when you try to read rows that haven't been calculated yet.
    # Thus, the majority of the time spent will usually be in categorize where the data is actually being
    # read.
    # Based on simple profiling, it looks like data normally starts to stream in fairly quickly, but
    # getting to the end takes awhile. So categorize gets its first records in a few seconds, but
    # runs for 10 times that long despite not doing much computation itself (i.e. it is blocking
    # waiting for bigquery).

    executor = ThreadPoolExecutor(2)
    basis_future = executor.submit(categorize, (basis_query))
    sample_future = executor.submit(categorize, (sample_query))
    basis_envs = basis_future.result()
    sample_envs = sample_future.result()

    conclusions_by_env = calculate_conclusions(basis_envs=basis_envs, sample_envs=sample_envs)

    context = {
        'basis_release': basis_release,
        'basis_start_dt': basis_start_dt,
        'basis_end_dt': basis_end_dt,
        'sample_release': sample_release,
        'sample_start_dt': sample_start_dt,
        'sample_end_dt': sample_end_dt,
    }

    if target_test_id:  # Rendering a specifically requested test id
        if not target_nurp_name:
            return HttpResponse(f'No nurp parameter was specified')

        samples_by_id, samples_by_component = sample_envs[target_nurp_name]
        basis_by_id, _ = basis_envs[target_nurp_name]

        if target_test_id not in samples_by_id:
            return HttpResponse(f'Target test {target_test_id} not found in nurp: {target_nurp_name}')

        sample_test_record = samples_by_id[target_test_id]
        basis_test_record = basis_by_id.get(target_test_id, TestRecord(test_name='', total_count=0, success_count=0, failure_count=0, test_id=target_test_id, flake_count=0))
        context['sample_test'] = sample_test_record
        context['basis_test'] = basis_test_record
        context['fishers_exact'] = str(fast_fisher.fast_fisher_cython.fisher_exact(
            sample_test_record.failure_count, sample_test_record.success_count,
            basis_test_record.failure_count, basis_test_record.success_count,
            alternative='greater'
        ))

        q = f'''
            SELECT prowjob_name, COUNT(*) as total_count, SUM(success_val) as success_count, SUM(flake_count) as flake_count 
            FROM `openshift-gce-devel.ci_analysis_us.junit` 
            WHERE modified_time >= DATETIME(TIMESTAMP "{basis_start_dt}:00+00") AND modified_time < DATETIME(TIMESTAMP "{basis_end_dt}:00+00") {target_nurp_filter} {target_test_filter}
                  AND branch = "{basis_release}" AND file_path NOT LIKE "%/junit_operator.xml" GROUP BY network, upgrade, arch, platform, prowjob_name        
        '''
        basis_prowjob_runs_rows = bq.query(q)

        q = f'''
            SELECT prowjob_name, COUNT(*) as total_count, SUM(success_val) as success_count, SUM(flake_count) as flake_count
            FROM `openshift-gce-devel.ci_analysis_us.junit` 
            WHERE modified_time >= DATETIME(TIMESTAMP "{sample_start_dt}:00+00") AND modified_time < DATETIME(TIMESTAMP "{sample_end_dt}:00+00") {target_nurp_filter} {target_test_filter}
                  AND branch = "{sample_release}" AND file_path NOT LIKE "%/junit_operator.xml" GROUP BY network, upgrade, arch, platform, prowjob_name        
        '''

        sample_prowjob_runs_rows = bq.query(q)

        # Aggregate test successes/failures by prowjob
        prowjob_names = set()
        basis_prowjob_runs: Dict[str, TestRecord] = dict()
        for row in basis_prowjob_runs_rows:
            prowjob_name: str = row['prowjob_name']
            prowjob_name = prowjob_name.replace(basis_release, 'X.X')  # Strip release specific information from prowjob name
            prowjob_names.add(prowjob_name)
            flake_count = row['flake_count']
            success_count = row['success_count']
            # In rare circumstances, based on the date range selected, it is possible for a failed test run to not be included
            # in the query while the success run (including a flake_count=1 reflecting the preceding, but un-selected
            # failure) is included. This could make total_count - flake_count a negative value.
            total_count = max(success_count, row['total_count'] - flake_count)
            basis_prowjob_runs[prowjob_name] = TestRecord(
                test_id=target_test_id,
                test_name=None,
                success_count=success_count,
                total_count=total_count,
                failure_count=total_count - success_count,
                flake_count=flake_count,
            )

        sample_prowjob_runs: Dict[str, TestRecord] = dict()
        for row in sample_prowjob_runs_rows:
            prowjob_name = row['prowjob_name']
            prowjob_name = prowjob_name.replace(sample_release, 'X.X')  # Strip release specific information from prowjob name
            prowjob_names.add(prowjob_name)
            flake_count = row['flake_count']
            success_count = row['success_count']
            # In rare circumstances, based on the date range selected, it is possible for a failed test run to not be included
            # in the query while the success run (including a flake_count=1 reflecting the preceding, but un-selected
            # failure) is included. This could make total_count - flake_count a negative value.
            total_count = max(success_count, row['total_count'] - flake_count)
            sample_prowjob_runs[prowjob_name] = TestRecord(
                test_id=target_test_id,
                test_name=None,
                success_count=success_count,
                total_count=total_count,
                failure_count=total_count - success_count,
                flake_count=flake_count,
            )

        prowjob_analysis: List[Dict[str, str]] = list()
        for prowjob_name in sorted(prowjob_names):
            basis_result = basis_prowjob_runs.get(prowjob_name, NO_TESTS)
            sample_result = sample_prowjob_runs.get(prowjob_name, NO_TESTS)

            if basis_result.total_count == 0 or sample_result.total_count == 0:
                regressed = False
            else:
                basis_pass_percentage = basis_result.success_count / basis_result.total_count
                sample_pass_percentage = sample_result.success_count / sample_result.total_count
                improved = sample_pass_percentage >= basis_pass_percentage

                regressed = fisher_significant(
                    sample_result.failure_count,
                    sample_result.success_count,
                    basis_result.failure_count,
                    basis_result.success_count,
                )

                if improved:
                    regressed = False

            basis_success_count = basis_result.success_count
            basis_failure_count = basis_result.failure_count
            basis_flake_count = basis_result.flake_count
            sample_success_count = sample_result.success_count
            sample_failure_count = sample_result.failure_count
            sample_flake_count = sample_result.flake_count

            prowjob_analysis.append(
                {
                    'prowjob_name': prowjob_name,
                    'basis_info': f'successes={basis_success_count} failures={basis_failure_count} (flakes={basis_flake_count})',
                    'sample_info': f'successes={sample_success_count} failures={sample_failure_count} (flakes={sample_flake_count})',
                    'individual_job_regression': ImageColumnLink(
                        image_path='/main/red.png' if regressed else '/main/green.png',
                        height=16, width=16,
                        href=None,
                        href_params=dict()
                    )
                }
            )

        prowjob_table = ProwjobTable(data=prowjob_analysis)
        context['table'] = prowjob_table
        context['breadcrumb'] = f'{target_nurp_name} > {target_component_name} > {target_capability_name} > {sample_test_record.test_name}'
        return render(request, 'main/report-test.html', context)

    if not target_component_name:  # Rendering all components
        component_summary: List[Dict] = list()
        extra_columns = []
        all_component_names = set()

        for nurp_name in sorted(conclusions_by_env.keys()):
            image_href_params = dict(context)
            image_href_params['nurp'] = nurp_name
            extra_columns.append((nurp_name, ImageColumn()))
            _, by_component = sample_envs[nurp_name]
            all_component_names.update(by_component.keys())

        for component_name in sorted(list(all_component_names)):
            if 'sig' not in component_name:
                continue

            row = {
                'name': component_name,
            }

            for nurp_name in sample_envs:
                _, samples_by_component = sample_envs[nurp_name]
                if component_name in samples_by_component:
                    regressed = samples_by_component[component_name].has_regressed(conclusions_by_env[nurp_name])
                else:
                    regressed = False

                href_params = dict(context)
                href_params['component'] = component_name
                href_params['nurp'] = nurp_name

                row[nurp_name] = ImageColumnLink(
                    image_path='/main/red.png' if regressed else '/main/green.png',
                    height=16, width=16,
                    href='/main/report',
                    href_params=dict(href_params)
                )

            component_summary.append(row)

        table = AllComponentsTable(component_summary,
                                   extra_columns=extra_columns,
                                   new_key='component',
                                   params=dict(context),
                                   )
        context['table'] = table
        context['breadcrumb'] = f'All Components'
        return render(request, 'main/report-table.html', context)
    else:  # Rendering all of a specific component's capabilities

        context['component'] = target_component_name

        if not target_nurp_name:  # Rendering a component or capability with nurps as column heading

            if target_capability_name:
                extra_columns = []

                for nurp_name in sorted(conclusions_by_env.keys()):
                    image_href_params = dict(context)
                    image_href_params['nurp'] = nurp_name
                    extra_columns.append((nurp_name, ImageColumn()))
                    _, by_component = sample_envs[nurp_name]

                test_summary: List[Dict] = list()

                for test_id, test_record in by_component[target_component_name].capabilities[target_capability_name].test_records.items():

                    row = {
                        'name': test_record.test_name,
                    }

                    for nurp_name in sample_envs:
                        regressed = has_regression([test_record], conclusions=conclusions_by_env[nurp_name])
                        href_params = dict(context)
                        href_params['capability'] = target_capability_name
                        href_params['test_id'] = test_id
                        href_params['nurp'] = nurp_name
                        row[nurp_name] = ImageColumnLink(
                            image_path='/main/red.png' if regressed else '/main/green.png',
                            height=16, width=16,
                            href='/main/report',
                            href_params=href_params,
                        )

                    test_summary.append(row)

                table = AllComponentsTable(test_summary,
                                           extra_columns=extra_columns,
                                           )
                context['table'] = table
                context['breadcrumb'] = f'{target_component_name} > {target_capability_name}'
                return render(request, 'main/report-table.html', context)

            else:
                extra_columns = []

                for nurp_name in sorted(conclusions_by_env.keys()):
                    image_href_params = dict(context)
                    image_href_params['nurp'] = nurp_name
                    extra_columns.append((nurp_name, ImageColumn()))
                    _, by_component = sample_envs[nurp_name]

                capability_summary: List[Dict] = list()

                for capability_name, capability_record in by_component[target_component_name].capabilities.items():

                    row = {
                        'name': capability_name,
                    }

                    for nurp_name in sample_envs:
                        regressed = capability_record.has_regressed(conclusions_by_env[nurp_name])
                        href_params = dict(context)
                        href_params['capability'] = capability_name
                        href_params['nurp'] = nurp_name
                        row[nurp_name] = ImageColumnLink(
                            image_path='/main/red.png' if regressed else '/main/green.png',
                            height=16, width=16,
                            href='/main/report',
                            href_params=href_params,
                        )

                    capability_summary.append(row)

                table = AllComponentsTable(capability_summary,
                                           extra_columns=extra_columns,
                                           new_key='capability',
                                           params=dict(context)
                                           )
                context['table'] = table
                context['breadcrumb'] = f'{target_component_name}'
                return render(request, 'main/report-table.html', context)

        else:
            samples_by_id, samples_by_component = sample_envs[target_nurp_name]
            if target_component_name and target_component_name not in samples_by_component:
                return HttpResponse(f'Component not found: {target_component_name}')

            context['nurp'] = target_nurp_name
            if not target_capability_name:
                capability_summary: List[Dict] = list()
                component_records = samples_by_component[target_component_name]
                for capability_name in sorted(component_records.capabilities.keys()):
                    regressed = component_records.capabilities[capability_name].has_regressed(conclusions_by_env[target_nurp_name])
                    href_params = dict(context)
                    href_params['capability'] = capability_name
                    capability_summary.append({
                        'name': capability_name,
                        'status': ImageColumnLink(
                            image_path='/main/red.png' if regressed else '/main/green.png',
                            height=16, width=16,
                            href='/main/report',
                            href_params=href_params,
                        )
                    })

                table = AllComponentsTable(capability_summary, extra_columns=[('status', ImageColumn())])
                context['table'] = table
                context['breadcrumb'] = f'{target_nurp_name} > {target_component_name}'
                return render(request, 'main/report-table.html', context)
            else:  # Rendering the capabilities of a specific component
                test_summary: List[Dict] = list()
                context['capability'] = target_capability_name
                component_records = samples_by_component[target_component_name]
                if target_capability_name not in component_records.capabilities:
                    return HttpResponse(f'Capability {target_capability_name} not found in component {target_component_name}')

                capability_records = component_records.capabilities[target_capability_name]
                for tr in sorted(list(capability_records.test_records.values()), key=lambda x: x.test_name):
                    regressed = has_regression([tr], conclusions_by_env[target_nurp_name])
                    href_params = dict(context)
                    href_params['test_id'] = tr.test_id
                    test_summary.append({
                        'name': tr.test_name,
                        'status': ImageColumnLink(
                            image_path='/main/red.png' if regressed else '/main/green.png',
                            height=16, width=16,
                            href='/main/report',
                            href_params=href_params,
                        )
                    })

                table = AllComponentsTable(test_summary, extra_columns=[('status', ImageColumn())])
                context['table'] = table
                context['breadcrumb'] = f'{target_nurp_name} > {target_component_name} > {target_capability_name}'
                return render(request, 'main/report-table.html', context)
