import numpy as np

def driverMaker(year1, year2):
    slope = np.loadtxt('../cq21/driver/cq21_slopef.txt', skiprows=6)
    # light = np.loadtxt(f'../cq21/driver/nl{year1}f.txt', skiprows=6)  # year1
    gdp = np.loadtxt(f'../cq21/driver/cq21_gdp{year1}.txt', skiprows=6)
    rainfall = np.loadtxt(f'../cq21/driver/railfall{year1}.txt', skiprows=6)
    dem = np.loadtxt(f'../cq21/driver/cq21_demf.txt', skiprows=6)
    # soildclass = np.loadtxt(f'../cq21/driver/soildclass.txt', skiprows=6)

    den_gov = np.loadtxt('../cq21/driver/dens_gov.txt', skiprows=6)
    den_highway = np.loadtxt('../cq21/driver/dens_tollsta.txt', skiprows=6)
    den_hospital = np.loadtxt('../cq21/driver/dens_hospi.txt', skiprows=6)
    den_mail = np.loadtxt('../cq21/driver/dens_mail.txt', skiprows=6)
    den_railway = np.loadtxt('../cq21/driver/dens_railway.txt', skiprows=6)
    den_school = np.loadtxt('../cq21/driver/dens_school.txt', skiprows=6)
    den_subway = np.loadtxt('../cq21/driver/dens_subway.txt', skiprows=6)
    # den_pois = np.loadtxt('../cq21/driver/den_allpois.txt', skiprows=6)

    dis_rail = np.loadtxt('../cq21/driver/dis_railwayf.txt', skiprows=6)
    # dis_road = np.loadtxt('../cq21/driver/dis_road.txt', skiprows=6)
    dis_road2 = np.loadtxt('../cq21/driver/dis_road2f.txt', skiprows=6)
    dis_road3 = np.loadtxt('../cq21/driver/dis_road3f.txt', skiprows=6)
    dis_road4 = np.loadtxt('../cq21/driver/dis_road4f.txt', skiprows=6)
    dis_road9 = np.loadtxt('../cq21/driver/dis_road9f.txt', skiprows=6)
    dis_highway = np.loadtxt('../cq21/driver/dis_roadhighf.txt', skiprows=6)
    dis_river = np.loadtxt('../cq21/driver/dis_waterf.txt', skiprows=6)
    # neighbor = np.loadtxt(f'../cq21/driver/neig92015.txt')
    # neighbor = np.loadtxt('../cq21/1520/data/neiProb_w3all.txt')
    # temp = np.zeros((6932147 - 6923420, neighbor.shape[1]))
    # neighbor = np.vstack((neighbor, temp))
    data = np.hstack(
        (slope.reshape(6932147, 1), gdp.reshape(6932147, 1),
         # light.reshape(6932147, 1),
         rainfall.reshape(6932147, 1), dem.reshape(6932147, 1),
         # soildclass.reshape(6932147, 1),
         den_gov.reshape(6932147, 1), den_highway.reshape(6932147, 1), den_hospital.reshape(6932147, 1),
         den_mail.reshape(6932147, 1), den_railway.reshape(6932147, 1), den_school.reshape(6932147, 1),
         den_subway.reshape(6932147, 1),
         # den_pois.reshape(6932147, 1),
         dis_road2.reshape(6932147, 1), dis_road3.reshape(6932147, 1), dis_road4.reshape(6932147, 1),
         dis_road9.reshape(6932147, 1), dis_highway.reshape(6932147, 1),
         dis_rail.reshape(6932147, 1),
         # dis_road.reshape(6932147, 1),
         dis_river.reshape(6932147, 1),
         # neighbor,
         ))
    print(data.shape)
    np.savetxt(f'../cq21/{year2}/data/new{year1}_driverAll_NoNei.txt', data, fmt='%f', delimiter=' ')  # year2
    return data


driverMaker(2015, 1520)

