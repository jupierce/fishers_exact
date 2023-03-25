from concurrent.futures import ThreadPoolExecutor

from typing import Dict, NamedTuple, List, Tuple, Any, Optional, Iterable, Set

from django.utils.html import format_html
from django.http import HttpResponse
import django_tables2 as tables

from .bq_junit import Junit, select, sum, count, any_value, EnvironmentModel, EnvironmentTestRecords, EnvironmentName, TestRecordAssessment, ComponentTestRecords, TestName, TestRecord, TestId

import fast_fisher.fast_fisher_cython


from django.shortcuts import render

ProwjobName = str


def index(request):
    return render(request, "main/index.html")


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


def _render_prowjob_rows(rows) -> str:
    result = ''
    for row in rows:
        if row['flake_count'] > 0:
            outcome_char = 's'
        elif row['success_count'] > 0:
            outcome_char = 'S'
        else:
            outcome_char = 'F'
        result += f'<a class="outcome_{outcome_char}" href="https://gcsweb-ci.apps.ci.l2s4.p1.openshiftapps.com/gcs/origin-ci-test/{row["file_path"]}">{outcome_char}</a> '
    return format_html(result)


class ProwjobTable(tables.Table):
    prowjob_name = tables.Column()
    basis_info = tables.Column()
    basis_runs = tables.Column()
    sample_info = tables.Column()
    sample_runs = tables.Column()

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
    target_test_uuid = insufficient_sanitization('test_uuid', None)
    target_platform_name = insufficient_sanitization('platform', None)
    target_upgrade_name = insufficient_sanitization('upgrade', None)
    target_arch_name = insufficient_sanitization('arch', None)
    target_network_name = insufficient_sanitization('network', None)
    group_by = insufficient_sanitization('group_by', None)
    target_environment_name = insufficient_sanitization('environment', None)

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

    if target_upgrade_name:
        pqb = pqb.filter(
            j.upgrade == target_upgrade_name
        )

    if target_arch_name:
        pqb = pqb.filter(
            j.arch == target_arch_name
        )

    if target_network_name:
        pqb = pqb.filter(
            j.network == target_network_name
        )

    if target_platform_name:
        pqb = pqb.filter(
            j.platform == target_platform_name
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
    basis_environment_model = EnvironmentModel()
    sample_environment_model = EnvironmentModel()
    basis_future = executor.submit(basis_environment_model.read_in_query, basis_query, group_by)
    sample_future = executor.submit(sample_environment_model.read_in_query, sample_query, group_by)
    basis_future.result()
    sample_future.result()
    sample_environment_model.build_mass_assessment_cache(basis_environment_model)

    ordered_environment_names: List[EnvironmentName] = sorted(list(sample_environment_model.get_ordered_environment_names()) + list(basis_environment_model.get_ordered_environment_names()))

    context = {
        'basis_release': basis_release,
        'basis_start_dt': basis_start_dt,
        'basis_end_dt': basis_end_dt,
        'sample_release': sample_release,
        'sample_start_dt': sample_start_dt,
        'sample_end_dt': sample_end_dt,
        'group_by': group_by,
    }

    if target_test_id:  # Rendering a specifically requested TestRecordSet test_id
        if not target_environment_name:
            return HttpResponse(f'No environment parameter was specified')
        if not target_component_name:
            return HttpResponse(f'No component parameter was specified')
        if not target_capability_name:
            return HttpResponse(f'No capability parameter was specified')

        sample_test_record_set = sample_environment_model.get_environment_test_records(target_environment_name).get_component_test_records(target_component_name).get_capability_test_records(target_capability_name).get_test_record_set(target_test_id)
        uuid_count = len(sample_test_record_set.test_records)
        if uuid_count > 1:
            # There are more than one test uuids that have been grouped into this test record set,
            # so provide a UI that allows the user to view each UUID and drill to the one they
            # want more information for.
            pass  # TODO: Implement
        elif uuid_count == 1:
            # There is only one test uuid associated with this test record set. Send the user
            # straight to the UI for the specific uuid.
            target_test_uuid = sample_test_record_set.get_test_record_uuids()[0]
        else:
            return HttpResponse('No tests are associated with this environment and test id')

    if target_test_uuid:
        if not target_environment_name:
            return HttpResponse(f'No environment parameter was specified')
        if not target_component_name:
            return HttpResponse(f'No component parameter was specified')
        if not target_capability_name:
            return HttpResponse(f'No capability parameter was specified')
        if not target_test_id:
            return HttpResponse(f'No test_id parameter was specified')

        sample_test_record_set = sample_environment_model.get_environment_test_records(
            target_environment_name).get_component_test_records(target_component_name).get_capability_test_records(
            target_capability_name).get_test_record_set(target_test_id)
        sample_test_record = sample_test_record_set.get_test_record(target_test_uuid)

        if not sample_test_record:
            return HttpResponse('No sample test record found for test uuid')

        basis_test_record_set = basis_environment_model.get_environment_test_records(
            target_environment_name).get_component_test_records(target_component_name).get_capability_test_records(
            target_capability_name).get_test_record_set(target_test_id)

        basis_test_record = basis_test_record_set.get_test_record(target_test_uuid)

        if not basis_test_record:
            basis_test_record = TestRecord(
                platform=sample_test_record.platform,
                network=sample_test_record.network,
                upgrade=sample_test_record.upgrade,
                arch=sample_test_record.arch,
                test_id=sample_test_record.test_id,
                test_name=sample_test_record.test_name
            )

        context['sample_test'] = sample_test_record
        context['basis_test'] = basis_test_record
        context['fishers_exact'] = str(fast_fisher.fast_fisher_cython.fisher_exact(
            sample_test_record.failure_count, sample_test_record.success_count,
            basis_test_record.failure_count, basis_test_record.success_count,
            alternative='greater'
        ))

        base_test_query = select(
            j.file_path,
            any_value('prowjob_name').label('prowjob_name'),
            sum(j.success_val).label('success_count'),
            sum(j.flake_count).label('flake_count'),
            count('*').label('total_count'),  # Including flakes
        ).where(
            j.platform == sample_test_record.platform,
            j.network == sample_test_record.network,
            j.upgrade == sample_test_record.upgrade,
            j.arch == sample_test_record.arch,
            j.test_id == sample_test_record.test_id
        ).group_by(
            j.file_path,
            j.modified_time
        ).order_by(
            j.modified_time  # Show test runs in roughly chronological order
        )

        basis_test_query = base_test_query.where(
            j.modified_time >= j.format_modified_time(basis_start_dt),
            j.modified_time < j.format_modified_time(basis_end_dt),
            j.branch == basis_release
        )

        sample_test_query = base_test_query.where(
            j.modified_time >= j.format_modified_time(sample_start_dt),
            j.modified_time < j.format_modified_time(sample_end_dt),
            j.branch == sample_release
        )

        # Aggregate test successes/failures by prowjob
        prowjob_names = set()

        basis_prowjob_runs: Dict[ProwjobName, List] = dict()
        for row in basis_test_query.execute():
            prowjob_name: str = row['prowjob_name']
            prowjob_name = prowjob_name.replace(basis_release, 'X.X')  # Strip release specific information from prowjob name
            prowjob_names.add(prowjob_name)
            if prowjob_name not in basis_prowjob_runs:
                basis_prowjob_runs[prowjob_name] = list()
            basis_prowjob_runs[prowjob_name].append(row)

        sample_prowjob_runs: Dict[ProwjobName, List] = dict()
        for row in sample_test_query.execute():
            prowjob_name: str = row['prowjob_name']
            prowjob_name = prowjob_name.replace(sample_release, 'X.X')  # Strip release specific information from prowjob name
            prowjob_names.add(prowjob_name)
            if prowjob_name not in sample_prowjob_runs:
                sample_prowjob_runs[prowjob_name] = list()
            sample_prowjob_runs[prowjob_name].append(row)

        prowjob_analysis: List[Dict[str, str]] = list()
        for prowjob_name in sorted(prowjob_names):

            basis_prowjob_rows = basis_prowjob_runs.get(prowjob_name, list())
            basis_success_count = 0
            basis_flake_count = 0
            basis_total_count = 0

            for basis_prowjob_row in basis_prowjob_rows:
                basis_success_count += basis_prowjob_row['success_count']
                basis_flake_count += basis_prowjob_row['flake_count']
                basis_total_count += basis_prowjob_row['total_count']  # Includes flakes
            basis_failure_count = max(0, basis_total_count-basis_flake_count-basis_success_count)

            sample_prowjob_rows = sample_prowjob_runs.get(prowjob_name, list())
            sample_success_count = 0
            sample_flake_count = 0
            sample_total_count = 0

            for sample_prowjob_row in sample_prowjob_rows:
                sample_success_count += sample_prowjob_row['success_count']
                sample_flake_count += sample_prowjob_row['flake_count']
                sample_total_count += sample_prowjob_row['total_count']  # Includes flakes
            sample_failure_count = max(0, sample_total_count-sample_flake_count-sample_success_count)

            prowjob_analysis.append(
                {
                    'prowjob_name': prowjob_name,
                    'basis_info': f'successes={basis_success_count} failures={basis_failure_count} (flakes={basis_flake_count})',
                    'basis_runs': _render_prowjob_rows(basis_prowjob_rows),
                    'sample_info': f'successes={sample_success_count} failures={sample_failure_count} (flakes={sample_flake_count})',
                    'sample_runs': _render_prowjob_rows(sample_prowjob_rows),
                }
            )

        prowjob_table = ProwjobTable(data=prowjob_analysis)
        context['table'] = prowjob_table
        context['breadcrumb'] = f'{target_environment_name} > {target_component_name} > {target_capability_name} > {sample_test_record.test_name}'
        return render(request, 'main/report-test.html', context)

    if not target_component_name:  # Rendering all components
        component_summary: List[Dict] = list()
        extra_columns = []
        all_component_names = set()

        for environment_name in ordered_environment_names:
            image_href_params = dict(context)
            image_href_params['environment'] = environment_name
            extra_columns.append((environment_name, ImageColumn()))
            sample_environment_test_records = sample_environment_model.get_environment_test_records(environment_name)
            all_component_names.update(sample_environment_test_records.get_component_names())

        for component_name in sorted(list(all_component_names)):
            if 'sig' not in component_name:
                continue

            row = {
                'name': component_name,
            }

            for environment_name in ordered_environment_names:
                sample_environment_test_records: EnvironmentTestRecords = sample_environment_model.get_environment_test_records(environment_name)
                sample_component_test_records = sample_environment_test_records.get_component_test_records(component_name)

                component_assessment: TestRecordAssessment = sample_component_test_records.assessment()

                href_params = dict(context)
                href_params['component'] = component_name
                href_params['environment'] = environment_name

                row[environment_name] = ImageColumnLink(
                    image_path=f'/main/{component_assessment.value}',
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

        if not target_environment_name:  # Rendering a component or capability with environments as column heading

            if target_capability_name:
                extra_columns = []

                test_id_lookup: Dict[TestName, TestId] = dict()
                for environment_name in sample_environment_model.get_ordered_environment_names():
                    image_href_params = dict(context)
                    image_href_params['environment'] = environment_name
                    extra_columns.append((environment_name, ImageColumn()))

                    # It's unlikely but possible that a component in one environment has different capabilities
                    # than a component in another environment. Build a full set of names across environments.
                    for sample_test_record_set in sample_environment_model.get_environment_test_records(environment_name).get_component_test_records(target_component_name).get_capability_test_records(target_capability_name).get_test_record_sets():
                        test_id_lookup[sample_test_record_set.canonical_test_name] = sample_test_record_set.test_id

                test_summary: List[Dict] = list()

                for test_name in sorted(list(test_id_lookup.keys())):
                    test_record_set_test_id = test_id_lookup[test_name]

                    row = {
                        'name': test_name,
                    }

                    for environment_name in sample_environment_model.get_ordered_environment_names():
                        assessment: TestRecordAssessment = sample_environment_model.get_environment_test_records(environment_name).get_component_test_records(target_component_name).get_capability_test_records(target_capability_name).get_test_record_set(test_record_set_test_id).assessment()
                        href_params = dict(context)
                        href_params['capability'] = target_capability_name
                        href_params['test_id'] = test_record_set_test_id
                        href_params['environment'] = environment_name
                        row[environment_name] = ImageColumnLink(
                            image_path=f'/main/{assessment.value}',
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

            else:  # Rending the capabilities of a component vs environment columns
                extra_columns = []

                # It's remote, but possible that the same component within different
                # environments have different capabilities. Develop a
                # set of all names across each environment.
                capability_names: Set[str] = set()

                for environment_name in sample_environment_model.get_ordered_environment_names():
                    image_href_params = dict(context)
                    image_href_params['environment'] = environment_name
                    extra_columns.append((environment_name, ImageColumn()))
                    capability_names.update(
                        sample_environment_model.get_environment_test_records(environment_name).get_component_test_records(target_component_name).get_capability_names()
                    )

                capability_summary: List[Dict] = list()
                for capability_name in sorted(capability_names):

                    row = {
                        'name': capability_name,
                    }

                    for environment_name in ordered_environment_names:
                        assessment = sample_environment_model.get_environment_test_records(environment_name).get_component_test_records(target_component_name).get_capability_test_records(capability_name).assessment()
                        href_params = dict(context)
                        href_params['capability'] = capability_name
                        href_params['environment'] = environment_name
                        row[environment_name] = ImageColumnLink(
                            image_path=f'/main/{assessment.value}',
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
            samples_by_id, samples_by_component = sample_environment_model[target_environment_name]
            if target_component_name and target_component_name not in samples_by_component:
                return HttpResponse(f'Component not found: {target_component_name}')

            context['environment'] = target_environment_name
            if not target_capability_name:
                capability_summary: List[Dict] = list()
                component_records = samples_by_component[target_component_name]
                for capability_name in sorted(component_records.capabilities.keys()):
                    regressed = component_records.capabilities[capability_name].has_regressed(conclusions_by_env[target_environment_name])
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
                context['breadcrumb'] = f'{target_environment_name} > {target_component_name}'
                return render(request, 'main/report-table.html', context)
            else:  # Rendering the capabilities of a specific component
                test_summary: List[Dict] = list()
                context['capability'] = target_capability_name
                component_records = samples_by_component[target_component_name]
                if target_capability_name not in component_records.capabilities:
                    return HttpResponse(f'Capability {target_capability_name} not found in component {target_component_name}')

                capability_records = component_records.capabilities[target_capability_name]
                for tr in sorted(list(capability_records.test_record_sets.values()), key=lambda x: x.test_name):
                    regressed = has_regression([tr], conclusions_by_env[target_environment_name])
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
                context['breadcrumb'] = f'{target_environment_name} > {target_component_name} > {target_capability_name}'
                return render(request, 'main/report-table.html', context)
