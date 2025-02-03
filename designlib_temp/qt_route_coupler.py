from typing import Tuple
import numpy as np
from copy import deepcopy
from qiskit_metal import draw, Dict
from designlib_temp.qt_component import QTComponent
from qiskit_metal.qlibrary.core import QComponent
from src.utils.utils import rotate_point, get_middle_point, get_normalized_vector 
from scipy.optimize import minimize

class InfoP:
    def __init__(
            self, 
            p,
            p_rot,
            p_fork,
            p_left_curve,
            p_right_curve,
            rotation_center,
            sgn,
            rotation_center_fork,
            angle,
            normal, 
            tangent,
            normal_rot,
            tangent_rot,
            length_arm,
            length_coupling,
            fork
            ):
        self.p = p
        self.p_rot = p_rot
        self.p_fork = p_fork
        self.p_left_curve= p_left_curve
        self.p_right_curve= p_right_curve
        self.rotation_center = rotation_center
        self.sgn = sgn
        self.rotation_center_fork = rotation_center_fork
        self.angle = angle
        self.normal = normal
        self.tangent = tangent
        self.normal_rot = normal_rot
        self.tangent_rot = tangent_rot
        self.length_arm = length_arm
        self.length_coupling = length_coupling
        self.fork = fork

class QTRouteCoupler(QComponent,QTComponent):
    """
    Inherits 'QComponent' and 'QTComponent' class.

    Description:
        A coupler, which can connect two pins either as two transmission lines coming in 
        parallel or one transmission line splitting into two and coming parallel to the 
        other transmission line.

    Options:
        * pin_inputs: takes the two pins.
        * coupling_length: length till which the arms overlap.
        * coupling_position: position of the overlap with respect to the default 
          overlap at the center.
        * conductor_width_inner: width of the first arm.
        * conductor_width_outer: width of the second arm.
        * conductor_separation: separation between the arms when they come in 
          parallel.
        * start_straight_inner: lenght from the pin till which the first arm 
          goes straight.
        * start_straight_outer: lenght from the pin till which the second arm 
          goes straight.
        * radius_inner_gap: radius of the bend the first arms takes to come in 
          parallel with the second arm. The gap and half the width of the first
          arm is added to it to ensure a proper turn.
        * gap_inner: gap of the first arm with the ground plane.
        * gap_outer: gap of the second arm with the ground plane.        
        * fork: whether one of the arm should fork. Defaults to False.
        * fork_flip: whether the second arm should fork or the first one. 
          Being True would lead to the forking of the second arm. Default to 
          False.
        * fork_separation: separation of the two new arms after forking of the 
          parent arm.
        * air_bridge (optional, defaults to False): Whether to add an air bridge at the starting or not.
        * air_bridge_length (optional, required when air_bridge=True): Width of the air bridge for the subtracted part.
        * air_bridge_width (optional, required when air_bridge=True): Length of the air bridge.
        * air_bridge_conductor_width (optional, required when air_bridge=True): Width of the air bridge for the conducting part.
        
    """

    default_options = Dict(
        pin_inputs=Dict(
            pin_inner=Dict(
                component='',  # Name of component connecting to conductor 1
                pin=''),  # Name of pin used for pin_1
            pin_outer=Dict(
                component='',  # Name of component connecting to conductor 2
                pin='')  # Name of pin used for pin_2
        ),
        coupling_length = '50um',
        coupling_position = '0um',
        conductor_width_inner = '20um', 
        conductor_width_outer = '20um',
        conductor_separation = '40um',
        start_straight_inner = '25um',
        start_straight_outer = '25um',
        radius_inner_gap = "100um",
        gap_inner = "60um",
        gap_outer = "60um",
        fork = False,
        fork_flip = False,
        fork_separation = '20um',
        air_bridge = False,
        air_bridge_length = '10um',
        air_bridge_conductor_width = '15um',
        air_bridge_width = '15um'
            )
    
    @staticmethod # make this static, send in all from pin, get InfoP, then run again a second time to get final bend and construct two curves
    def optimize_angles(
        p, 
        p_middle_inner,
        p_middle_outer,
        tangent_inner,
        tangent_outer,
        normal_inner,
        normal_outer,
        use_analytical:bool,
        ) -> Tuple[InfoP, InfoP]:
        """
        The coupler needs the possibility to bend twice to fulfill 
        1) paralell lines at 
        2) requested distance.

        first bend is calculated *analytically* to make the lines parallel
        second bend is calculated *numerically* to make the lines at the requested distance
        """

        vec_in_to_out = p_middle_outer - p_middle_inner
        
        if p.fork:
            target_distance = 0
        else:
            target_distance = p.conductor_width_inner/2 + p.conductor_separation + p.conductor_width_outer/2

        radius_inner = p.radius_inner_gap + p.gap_inner + p.conductor_width_inner/2
        if use_analytical:
            sgn_rot_center_inner = np.sign(np.dot(vec_in_to_out, tangent_inner))
            sgn_rot_center_outer = np.sign(np.dot(-vec_in_to_out, tangent_outer))
            radius_outer = p.radius_inner_gap + p.gap_inner + p.conductor_width_inner + p.conductor_separation + p.conductor_width_outer/2
            # print("sgn_rot_center_inner", sgn_rot_center_inner, "sgn_rot_center_outer", sgn_rot_center_outer)
        else:
            vec_inner_to_outer = p_middle_outer - p_middle_inner
            sgn_rot_center_inner = -np.sign(target_distance - np.dot(tangent_inner, vec_inner_to_outer))
            sgn_rot_center_outer = sgn_rot_center_inner
            radius_outer = radius_inner



        rotation_center_inner = p_middle_inner + sgn_rot_center_inner * tangent_inner * radius_inner
        rotation_center_outer = p_middle_outer + sgn_rot_center_outer * tangent_outer * radius_outer
        
        # Initial guess
        def angle_between_vectors(v1, v2):
            """
            Calculate the angle in radians between vectors 'v1' and 'v2'
            """
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        def func_to_minimize(angle: np.ndarray):
            angle = angle[0]
            p_rot_inner = rotate_point(point = p_middle_inner, rotation_center = rotation_center_inner, angle_rad = angle)
            p_rot_outer = rotate_point(point = p_middle_outer, rotation_center = rotation_center_outer, angle_rad = angle)
            tangent_rot_inner = rotate_point(point = tangent_inner, rotation_center = (0,0), angle_rad = angle)
            tangent_rot_outer = rotate_point(point = tangent_outer, rotation_center = (0,0), angle_rad = angle)

            normal_rot_inner  = rotate_point(point = normal_inner,  rotation_center = (0,0), angle_rad = angle)
            normal_rot_outer  = rotate_point(point = normal_outer,  rotation_center = (0,0), angle_rad = angle)
            
            vec_inner_to_outer_rot = p_rot_outer - p_rot_inner
            distance_at_inner = np.dot(sgn_rot_center_inner * tangent_rot_inner, -vec_inner_to_outer_rot) # TODO why is the sign here?
            distance_at_outer = np.dot(sgn_rot_center_outer * tangent_rot_outer, vec_inner_to_outer_rot)
        
            regularization = 1e-6
            cost_distance_between_conductor_at_inner = (target_distance-distance_at_inner)**2/(target_distance**2 + regularization)
            cost_distance_between_conductor_at_outer = (target_distance-distance_at_outer)**2/(target_distance**2 + regularization)
            cost_conductors_are_parallel = (np.dot(normal_rot_inner, normal_rot_outer) + 1)**2
            
            total_cost = cost_distance_between_conductor_at_inner + cost_distance_between_conductor_at_outer + cost_conductors_are_parallel
            # print("target_distance", target_distance, "distance_at_inner", distance_at_inner, "distance_at_outer", distance_at_outer, )
            # print("total_cost", total_cost, "cost_inner", cost_distance_between_conductor_at_inner, "cost_outer", cost_distance_between_conductor_at_outer, "cost_conductors_are_parallel", cost_conductors_are_parallel)
            return total_cost
        
        if use_analytical:
            angle_inner = np.sign(np.dot(vec_in_to_out, tangent_inner)) * angle_between_vectors(vec_in_to_out, normal_inner)
            angle_outer = np.sign(np.dot(-vec_in_to_out, tangent_outer)) * angle_between_vectors(-vec_in_to_out, normal_outer)
        else:
            x0 = [sgn_rot_center_inner * 10 / 180 * np.pi]
            res = minimize(func_to_minimize, x0)
            angle_inner =  res.x[0]
            angle_outer = angle_inner 
   
        p_rot_inner = rotate_point(point = p_middle_inner, rotation_center = rotation_center_inner, angle_rad = angle_inner)
        p_rot_outer = rotate_point(point = p_middle_outer, rotation_center = rotation_center_outer, angle_rad = angle_outer)
        tangent_rot_inner = rotate_point(point = tangent_inner, rotation_center = (0,0), angle_rad = angle_inner)
        tangent_rot_outer = rotate_point(point = tangent_outer, rotation_center = (0,0), angle_rad = angle_outer)
        normal_rot_inner = rotate_point(point = normal_inner, rotation_center = (0,0), angle_rad = angle_inner)
        normal_rot_outer = rotate_point(point = normal_outer, rotation_center = (0,0), angle_rad = angle_outer)
        length = abs(np.dot(sgn_rot_center_inner * normal_rot_inner, p_rot_outer - p_rot_inner))
        fork_inner = True if p.fork else False
        fork_outer = False
        if p.fork_flip:
            fork_inner = not fork_inner
            fork_outer = not fork_outer
        gap_fork_inner = p.fork_separation + p.conductor_width_inner
        gap_fork_outer = p.fork_separation + p.conductor_width_outer
        length_inner = (length/2 - gap_fork_inner - p.coupling_length/2 if p.fork else length/2 + p.coupling_length/2) + p.coupling_position
        length_outer = (length/2 + p.coupling_length/2 if p.fork else length/2 + p.coupling_length/2) - p.coupling_position
        p_arm_inner = p_rot_inner + normal_rot_inner*(length_inner)
        p_arm_outer = p_rot_outer + normal_rot_outer*(length_outer)
        p_fork_inner = p_arm_inner - normal_rot_inner*(p.conductor_width_inner/2)
        p_fork_outer = p_arm_outer - normal_rot_outer*(p.conductor_width_outer/2)
        rotation_center_fork_inner = p_arm_inner + normal_rot_inner*(gap_fork_inner)
        rotation_center_fork_outer = p_arm_outer + normal_rot_outer*(gap_fork_outer)
        p_left_curve_inner = rotate_point(point=p_fork_inner, rotation_center = rotation_center_fork_inner, angle_rad=-90*np.pi/180)
        p_right_curve_inner = rotate_point(point=p_fork_inner, rotation_center = rotation_center_fork_inner, angle_rad=90*np.pi/180)
        p_left_curve_outer = rotate_point(point=p_fork_outer, rotation_center=rotation_center_fork_outer, angle_rad=-90*np.pi/180)
        p_right_curve_outer = rotate_point(point=p_fork_outer, rotation_center=rotation_center_fork_outer, angle_rad=90*np.pi/180)

        inner_info = InfoP(
            p_middle_inner, 
            p_rot_inner,
            p_fork_inner,
            p_left_curve_inner,
            p_right_curve_inner,
            rotation_center_inner, 
            sgn_rot_center_inner,
            rotation_center_fork_inner,
            angle_inner, 
            normal_inner,
            tangent_inner,
            normal_rot_inner, 
            tangent_rot_inner, 
            length_inner,
            p.coupling_length,
            fork_inner)
        
        outer_info = InfoP(
            p_middle_outer, 
            p_rot_outer, 
            p_fork_outer,
            p_left_curve_outer,
            p_right_curve_outer,
            rotation_center_outer, 
            sgn_rot_center_outer,
            rotation_center_fork_outer,
            angle_outer, 
            normal_outer,
            tangent_outer,
            normal_rot_outer, 
            tangent_rot_outer, 
            length_outer,
            p.coupling_length,
            fork_outer)
        
        return inner_info, outer_info

    @staticmethod
    def get_rect(point, tangent, normal, width, length, populate_rect=None): 
        """Creates all the straight parts of the coupler"""
        left = point + tangent * width / 2
        right = point - tangent * width / 2
        top_left = left + normal * length
        top_right = right + normal * length

        
        coords_left = [left,top_left]
        coords_right = [right,top_right]
        coords = [left, right, top_right, top_left]

        if populate_rect is not None:
            populate_rect.extend(coords)

        return coords_left,coords_right
    
    @staticmethod
    def get_bend(point, rotation_center, sgn, angle, width: float): #creates all the circular/bent part of the coupler
        if (rotation_center == point).all():
            return draw.Polygon([]) # TODO this might be wrong
        else:
            nbr_points = max(int(360 / (2*np.pi) * np.abs(angle) / 4), 2)
            angles = np.linspace(0, angle, num=nbr_points, endpoint=True)  # Increase num for a smoother arc
            center = (rotation_center-point)/np.linalg.norm(rotation_center-point)
            p_inner = point + sgn * center* min(width / 2,np.linalg.norm(rotation_center-point))
            p_outer = point - sgn * center* width / 2
            arc_points_left  = [rotate_point(p_inner, rotation_center, angle) for angle in angles]
            arc_points_right = [rotate_point(p_outer, rotation_center, angle) for angle in angles]

            return arc_points_left, arc_points_right
    
    @staticmethod
    def get_min_index(list,point):
        list = list - point
        list = np.diag(np.dot(list,list.transpose()))
        return np.where(list==min(list))[0][0]
    
    def get_min_list(self,list,point1,point2):
        index1 = self.get_min_index(list,point1)
        index2 = self.get_min_index(list,point2)
        if index1<index2:
            try:
                return list[index1:(index2+1)]
            except:
                return list[index1:]
        else:
            return np.array([*list[index1:],*list[:(index2+1)]])

    def air_bridge(self,geometry,mid_point,normal,tangent,arm_width,center_width,air_bridge_width,air_bridge_start_straight,populate_air_bridge_position:bool):
        points = Dict(
        point = mid_point,
        normal = normal,
        tangent = tangent,
        width = arm_width*1 # Use 1.01 this the cuts are not proper
        )

        if arm_width<= center_width:
            return geometry

        if air_bridge_width>air_bridge_start_straight:
            raise Exception("For an air bridge for the component",self.name, "air_bridge_width: ",air_bridge_width," mm, cannot be larger than start_straight:",air_bridge_start_straight," mm.")
            
        temp = self.get_rect(points.point,points.tangent,points.normal,points.width,air_bridge_start_straight)
        rect = draw.Polygon([*temp[0],*(temp[1][::-1])])

        points.point = deepcopy(mid_point)
        points.width = deepcopy(arm_width)
        air_bridge_start_length = (air_bridge_start_straight-air_bridge_width)/2
        rect_air_bridge_inner = []
        rect_air_bridge_outer = []
        rect_left_of_narrow_air_bridge = self.get_rect(points.point,points.tangent,points.normal,points.width,air_bridge_start_length)
        rect_air_bridge_inner+= rect_left_of_narrow_air_bridge[0]
        rect_air_bridge_outer+= rect_left_of_narrow_air_bridge[1]
        points.point += points.normal*air_bridge_start_length
        points.width = deepcopy(center_width)
        rect_narrow_air_bridge = self.get_rect(points.point,points.tangent,points.normal,points.width,air_bridge_width)
        rect_air_bridge_anchored = self.get_rect(points.point,points.tangent,points.normal,points.width+air_bridge_width,air_bridge_width)
        if populate_air_bridge_position:
            x_start = get_middle_point(rect_air_bridge_anchored[0][0], rect_air_bridge_anchored[0][1])[0]
            y_start = get_middle_point(rect_air_bridge_anchored[0][0], rect_air_bridge_anchored[0][1])[1]
            z_start = 0
            x_end = get_middle_point(rect_air_bridge_anchored[1][0], rect_air_bridge_anchored[1][1])[0]
            y_end = get_middle_point(rect_air_bridge_anchored[1][0], rect_air_bridge_anchored[1][1])[1]
            z_end = 0
            self.add_air_bridge_coordinates([[x_start,y_start,z_start], [x_end,y_end,z_end]])

        rect_air_bridge_inner+= rect_narrow_air_bridge[0]
        rect_air_bridge_outer+= rect_narrow_air_bridge[1]
        points.point += points.normal*air_bridge_width
        points.width = deepcopy(arm_width)
        rect_right_of_narrow_air_bridge = self.get_rect(points.point,points.tangent,points.normal,points.width,air_bridge_start_length)
        rect_air_bridge_inner+= rect_right_of_narrow_air_bridge[0]
        rect_air_bridge_outer+= rect_right_of_narrow_air_bridge[1]

        rect_air_bridge = draw.Polygon([*rect_air_bridge_inner,*(rect_air_bridge_outer[::-1])])
        
        rounding = min(air_bridge_start_length,air_bridge_width,center_width)/2.1
        rect_air_bridge = rect_air_bridge.buffer(-rounding).buffer(2*rounding).buffer(-rounding)
    
        points.point = deepcopy(mid_point)
        temp = self.get_rect(points.point,points.tangent,points.normal,points.width,air_bridge_start_length/2)
        rect1 = draw.Polygon([*temp[0],*(temp[1][::-1])])
        points.point += points.normal*air_bridge_start_straight*1.001 # Use 1.01 if the joining is not proper
        temp = self.get_rect(points.point,points.tangent,points.normal,points.width,-air_bridge_start_length/2)
        rect2 = draw.Polygon([*temp[0],*(temp[1][::-1])])

        rect_air_bridge = draw.subtract(rect_air_bridge,rect1.buffer(0.001))
        rect_air_bridge = draw.subtract(rect_air_bridge,rect2.buffer(0.001))

        plist = draw.get_poly_pts(rect_air_bridge)
        list1 = self.get_min_list(plist,draw.get_poly_pts(rect)[0],draw.get_poly_pts(rect)[1])
        list2 = self.get_min_list(plist,draw.get_poly_pts(rect)[2],draw.get_poly_pts(rect)[3])

        geometry[0][1:1] = (list1).tolist()
        geometry[1][1:1] = (list2[::-1]).tolist()

        return geometry
    
    def get_curve( 
            self,
            p_info_1: InfoP,
            p_info_2: InfoP,
            width: float, 
            start_straight: float,
            air_bridge_center_width: float,
            air_bridge_width: float,
            exclude_continue_straight: bool = False,
            length_gap: float = 0, #This is an added length so that the pins have correct separation with ground
            populate_rect: dict = None, # used to calculate center of coupler, e.g. to be used for positioning charge line
            populate_air_bridge_position: bool = False,
            ):
            """The parent function that creates all the shapely objects for the coupler"""
            p = self.parse_options()
            p1 = p_info_1
            p2 = p_info_2
            p_inner = []
            p_outer = []
            start_straight_left,start_straight_right = self.get_rect(p1.p, p1.tangent, -p1.normal, width, start_straight) #initial straight extension from the pin
            p_inner+=start_straight_left[::-1]
            p_outer+=start_straight_right[::-1]
            bend_1_left, bend_1_right = self.get_bend(p1.p, p1.rotation_center, p1.sgn, p1.angle, width) #bend to make the two arms parallel
            p_inner+=bend_1_left
            p_outer+=bend_1_right
            bend_2_left, bend_2_right = self.get_bend(p2.p, p2.rotation_center, p2.sgn, p2.angle, width) #bend to make the two arms parallel
            p_inner+=bend_2_left
            p_outer+=bend_2_right
            extra_coverage = width/2 #min(width,(p2.length_arm + length_gap)/4) # The purpose of this extra coverage is as follows. In the continue straight part, we want there to be rounding at the end.
            # But, due to this rounding, there might be sharp points where the continue straight joins the bends/curves. Therefore, we want there to be an extended straight region (this region will not be rounded) 
            # which overlaps with the rounded region of the continue straight, so that it covers it and there are no sharp points. This overlap should be width/2. An assumption goes here that length + length_gap > width/2.
            # I think this assumption will almost always be true.
            if extra_coverage>0:
                continue_half_left,continue_half_right = self.get_rect(p2.p_rot, p2.tangent_rot, p2.normal_rot, width, extra_coverage, populate_rect=populate_rect) #edit
                p_inner+=continue_half_left
                p_outer+=continue_half_right
            
            
            if p.air_bridge:
                [p_inner,p_outer] = self.air_bridge([p_inner,p_outer],p1.p -p1.normal*start_straight,p1.normal,p1.tangent,width,air_bridge_center_width,air_bridge_width,start_straight,populate_air_bridge_position=populate_air_bridge_position)

            curve1 = draw.Polygon([*p_inner,*(p_outer[::-1])])

            if exclude_continue_straight:
                return curve1
            
            continue_straight_left,continue_straight_right = self.get_rect(p2.p_rot , p2.tangent_rot, p2.normal_rot, width, p2.length_arm + length_gap , populate_rect=populate_rect) #parallel straight arm after the bend
            continue_straight = draw.Polygon([*continue_straight_left,*(continue_straight_right[::-1])])
            curve2 = draw.union([continue_straight])

            if not p2.fork:
                return curve2

            uleft_left, uleft_right = self.get_bend(p2.p_fork, p2.rotation_center_fork, p2.sgn, -90*np.pi/180, width) #forking
            uleft = draw.Polygon([*uleft_left,*(uleft_right[::-1])])
            uright_left, uright_right = self.get_bend(p2.p_fork, p2.rotation_center_fork, p2.sgn, 90*np.pi/180, width)
            uright = draw.Polygon([*uright_left,*(uright_right[::-1])])
            curve2 = draw.union([curve2,uleft,uright])

            fork_left_left,fork_left_right = self.get_rect(p2.p_left_curve, p2.tangent_rot, p2.normal_rot, width, p2.length_coupling + length_gap) #parallel straight arm after forking
            fork_left = draw.Polygon([*fork_left_left,*(fork_left_right[::-1])])
            fork_right_left,fork_right_right = self.get_rect(p2.p_right_curve, p2.tangent_rot, p2.normal_rot, width, p2.length_coupling + length_gap)
            fork_right = draw.Polygon([*fork_right_left,*(fork_right_right[::-1])])
            curve2 = draw.union([curve2,fork_left,fork_right])

            return curve2
    @staticmethod
    def force_right_hand_side_coord_system(pin):
        """Forces the convention that the tangent should be to the left hand side to the normal"""
        if np.dot(rotate_point(pin.normal, [0,0], np.pi/2), pin.tangent) < 0.99:
            pin.tangent = -pin.tangent
    
    def set_pin(self, name: str):
        """Defines the CPW pins and returns the pin coordinates and normal
        direction vector.

        Args:
            name: String (supported pin names are: start, end)

        Return:
            QRoutePoint: Last point (for now the single point) in the QRouteLead

        Raises:
            Exception: Ping name is not supported
        """
        # First define which pin/lead you intend to initialize
        if name == self.inner_pin_name:
            options_pin = self.options.pin_inputs.pin_inner
            width = self.p.conductor_width_inner
        elif name == self.outer_pin_name:
            options_pin = self.options.pin_inputs.pin_outer
            width = self.p.conductor_width_outer

        reference_pin = deepcopy(self.get_connected_pin(options_pin))
        pin_points = [reference_pin.middle+reference_pin.normal, reference_pin.middle]
        self.add_pin(name, pin_points, width, input_as_norm=True)
        self.force_right_hand_side_coord_system(self.pins[name])

        self.design.connect_pins(
            self.design.components[options_pin.component].id, options_pin.pin,
            self.id, name)
        self.calls += 1
    
    def make(self):
        self.calls = 0
        p = self.p
        self.inner_pin_name = "inner"
        self.outer_pin_name = "outer"

        self.pin_inner = deepcopy(self.get_connected_pin(p.pin_inputs['pin_inner']))
        self.pin_outer = deepcopy(self.get_connected_pin(p.pin_inputs['pin_outer']))
        self.force_right_hand_side_coord_system(self.pin_inner)
        self.force_right_hand_side_coord_system(self.pin_outer)

        self.set_pin(self.inner_pin_name)
        self.set_pin(self.outer_pin_name)

        inner_info_1, outer_info_1 = self.optimize_angles(
            self.p,
            self.pin_inner.middle + self.pin_inner.normal * self.p.start_straight_inner,
            self.pin_outer.middle + self.pin_outer.normal * self.p.start_straight_outer,
            self.pin_inner.tangent,
            self.pin_outer.tangent,
            self.pin_inner.normal,
            self.pin_outer.normal,
            use_analytical=True
        )
        inner_info_2, outer_info_2 = self.optimize_angles(
            self.p,
            inner_info_1.p_rot,
            outer_info_1.p_rot,
            inner_info_1.tangent_rot,
            outer_info_1.tangent_rot,
            inner_info_1.normal_rot,
            outer_info_1.normal_rot,
            use_analytical=False
        )
        
        rect_1 = []
        rect_2 = []

        curve_inner = self.get_curve(inner_info_1, inner_info_2, p.conductor_width_inner, p.start_straight_inner,air_bridge_center_width = p.air_bridge_conductor_width, air_bridge_width = p.air_bridge_width, populate_rect=rect_1)
        curve_outer = self.get_curve(outer_info_1, outer_info_2, p.conductor_width_outer, p.start_straight_outer,air_bridge_center_width = p.air_bridge_conductor_width, air_bridge_width = p.air_bridge_width,  populate_rect=rect_2)
        curve_inner_round = self.round_object(curve_inner, p.conductor_width_inner / 2.01)
        curve_outer_round = self.round_object(curve_outer, p.conductor_width_outer / 2.01)
        self._populate_center_point_of_coupler(rect_1, rect_2)

        sharp_corners_inner = self.get_curve(inner_info_1, inner_info_2, p.conductor_width_inner, p.start_straight_inner,air_bridge_center_width = p.air_bridge_conductor_width, air_bridge_width = p.air_bridge_width, exclude_continue_straight=True)
        sharp_corners_outer = self.get_curve(outer_info_1, outer_info_2, p.conductor_width_outer, p.start_straight_outer,air_bridge_center_width = p.air_bridge_conductor_width, air_bridge_width = p.air_bridge_width, exclude_continue_straight=True)
        
        curve_inner_round_cornered = draw.union([curve_inner_round, sharp_corners_inner])
        curve_outer_round_cornered = draw.union([curve_outer_round, sharp_corners_outer])

        curve_inner_gap = self.get_curve(inner_info_1, inner_info_2, 2*p.gap_inner + p.conductor_width_inner, p.start_straight_inner,air_bridge_center_width = p.air_bridge_length, air_bridge_width = p.air_bridge_width,length_gap = p.gap_inner, populate_air_bridge_position=True)
        curve_outer_gap = self.get_curve(outer_info_1, outer_info_2, 2*p.gap_outer + p.conductor_width_outer, p.start_straight_outer,air_bridge_center_width = p.air_bridge_length, air_bridge_width = p.air_bridge_width,length_gap = p.gap_outer, populate_air_bridge_position=True)
        both_gaps = draw.union([curve_inner_gap, curve_outer_gap])

        sharp_corners_gap_inner = self.get_curve(inner_info_1, inner_info_2, 2*p.gap_inner + p.conductor_width_inner, p.start_straight_inner,air_bridge_center_width = p.air_bridge_length, air_bridge_width = p.air_bridge_width, exclude_continue_straight=True)
        sharp_corners_gap_outer = self.get_curve(outer_info_1, outer_info_2, 2*p.gap_outer + p.conductor_width_outer, p.start_straight_outer,air_bridge_center_width = p.air_bridge_length, air_bridge_width = p.air_bridge_width, exclude_continue_straight=True)

        rounding_gap = min(p.conductor_width_inner + 2*p.gap_inner, p.conductor_width_outer + 2*p.gap_outer) / 2.01
        curve_inner_round = self.round_object(both_gaps, rounding_gap)
        gap_round_cornered = draw.union([curve_inner_round, sharp_corners_gap_inner, sharp_corners_gap_outer])
        
        gap_round_cornered_merged = draw.union([
            curve_inner_round_cornered, curve_outer_round_cornered
        ])

        # Add all geometries to qgeometry
        if p.conductor_separation == -p.conductor_width_inner:
            # Conductors may not overlap in capacitance simulation
            self.add_qgeometry('poly', {'curve': gap_round_cornered_merged}, chip=p.chip)
        else:
            self.add_qgeometry('poly', {'curve_inner': curve_inner_round_cornered, 'curve_outer': curve_outer_round_cornered}, chip=p.chip)

        self.add_qgeometry('poly', {'curve_gap': gap_round_cornered}, chip=p.chip, subtract = True)

    def _populate_center_point_of_coupler(self, rect_1, rect_2):
        middle_1 = get_middle_point(rect_1[2], rect_1[3])
        middle_2 = get_middle_point(rect_2[2], rect_2[3])
        self.center_point_of_coupler = get_middle_point(middle_1, middle_2)
        self.normal_vector_of_coupler = get_normalized_vector(rect_1[0], rect_1[1])
        
    def get_center_point_of_coupler(self):
        return self.center_point_of_coupler
    
    def get_normal_vector_of_coupler(self):
        return self.normal_vector_of_coupler
    
    def get_rotation_of_coupler(self):
        """ Orientation angle in degrees"""
        return 180 / np.pi * np.arctan2(self.normal_vector_of_coupler[1], self.normal_vector_of_coupler[0])