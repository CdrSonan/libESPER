/*
 * Copyright (c) 2002, 2017 Jens Keiner, Stefan Kunis, Daniel Potts
 *
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#include "infft.h"

/* Coefficients for Lanzcos's approximation to the Gamma function. Can be
 * regenerated with Mathematica from file lambda.nb. */

#if defined(NFFT_LDOUBLE)
  #if LDBL_MANT_DIG > 64
    /* long double 128 bit wide */
    #define N 24
    static const R num[24] =
    {
      K(3.035162425359883494754028782232869726547E21),
      K(3.4967568944064301036001605717507506346E21),
      K(1.9266526566893208886540195401514595829E21),
      K(6.755170664882727663160830237424406199E20),
      K(1.691728531049187527800862627495648317E20),
      K(3.21979351672256057856444116302160246E19),
      K(4.8378495427140832493758744745481812E18),
      K(5.8843103809049324230843820398664955E17),
      K(5.893958514163405862064178891925630E16),
      K(4.919561837722192829918665308020810E15),
      K(3.449165802442404074427531228315120E14),
      K(2.041330296068782505988459692384726E13),
      K(1.022234822943784007524609706893119E12),
      K(4.33137871919821354846952908076307E10),
      K(1.54921950559667418528481770869280E9),
      K(4.6544421199876191938054157935810E7),
      K(1.16527806807504975090675074910053E6),
      K(24024.759267256769471083727721827),
      K(400.96500811342195582435806376976),
      K(5.2829901565447826961703902917085),
      K(0.05289990244125101024092566765994),
      K(0.0003783467106547406854542665695934),
      K(1.7219414217921113919596660801124E-6),
      K(3.747999317071488557713812635427084359354E-9)
    };
/*    static const R denom[24] =
    {
      K(0.0),
      K(1124000727777607680000.0),
      K(4148476779335454720000.0),
      K(6756146673770930688000.0),
      K(6548684852703068697600.0),
      K(4280722865357147142912.0),
      K(2021687376910682741568.0),
      K(720308216440924653696.0),
      K(199321978221066137360.0),
      K(43714229649594412832.0),
      K(7707401101297361068.0),
      K(1103230881185949736.0),
      K(129006659818331295.0),
      K(12363045847086207.0),
      K(971250460939913.0),
      K(62382416421941.0),
      K(3256091103430.0),
      K(136717357942.0),
      K(4546047198.0),
      K(116896626.0),
      K(2240315.0),
      K(30107.0),
      K(253.0),
      K(1.0L)
    };*/
    static const R g = K(20.32098218798637390136718750000000000000);
  #elif LDBL_MANT_DIG == 64
    /* long double 96 bit wide */
    #define N 17
    static const R num[17] =
    {
      K(2.715894658327717377557655133124376674911E12),
      K(3.59017952609791210503852552872112955043E12),
      K(2.22396659973781496931212735323581871017E12),
      K(8.5694083451895624818099258668254858834E11),
      K(2.2988587166874907293359744645339939547E11),
      K(4.552617168754610815813502794395753410E10),
      K(6.884887713165178784550917647709216425E9),
      K(8.11048596140753186476028245385237278E8),
      K(7.52139159654082231449961362311950170E7),
      K(5.50924541722426515169752795795495283E6),
      K(317673.536843541912671493184218236957),
      K(14268.2798984503552014701437332033752),
      K(489.361872040326367021390908360178781),
      K(12.3894133003845444929588321786545861),
      K(0.218362738950461496394157450728168315),
      K(0.00239374952205844918669062799606398310),
      K(0.00001229541408909435212800785616808830746135)
    };
/*    static const R denom[17] =
    {
      K(0.0),
      K(1307674368000.0),
      K(4339163001600.0),
      K(6165817614720.0),
      K(5056995703824.0),
      K(2706813345600.0),
      K(1009672107080.0),
      K(272803210680.0),
      K(54631129553.0),
      K(8207628000.0),
      K(928095740.0),
      K(78558480.0),
      K(4899622.0),
      K(218400.0),
      K(6580.0),
      K(120.0),
      K(1.0L)
    };*/
    static const R g = K(12.22522273659706115722656250000000000000);
  #else
    #error Unsupported size of long double
  #endif
#elif defined(NFFT_SINGLE)
  /* float */
  #define N 6
  static const R num[6] =
  {
    K(14.02614328749964766195705772850038393570),
    K(43.74732405540314316089531289293124360129),
    K(50.59547402616588964511581430025589038612),
    K(26.90456680562548195593733429204228910299),
    K(6.595765571169314946316366571954421695196),
    K(0.6007854010515290065101128585795542383721)
  };
/*  static const R denom[6] =
  {
    K(0.0),
    K(24.0),
    K(50.0),
    K(35.0),
    K(10.0),
    K(1.0)
  };*/
  static const R g = K(1.428456135094165802001953125000000000000);
#else
  /* double */
  #define N 13
  static const R num[13] =
  {
    K(5.690652191347156388090791033559122686859E7),
    K(1.037940431163445451906271053616070238554E8),
    K(8.63631312881385914554692728897786842234E7),
    K(4.33388893246761383477372374059053331609E7),
    K(1.46055780876850680841416998279135921857E7),
    K(3.48171215498064590882071018964774556468E6),
    K(601859.61716810987866702265336993523025),
    K(75999.293040145426498753034435989091371),
    K(6955.9996025153761403563101155151989875),
    K(449.944556906316811944685860765098840962),
    K(19.5199278824761748284786096623565213621),
    K(0.509841665565667618812517864480469450999),
    K(0.006061842346248906525783753964555936883222)
  };
/*  static const R denom[13] =
  {
    K(0.0),
    K(39916800.0),
    K(120543840.0),
    K(150917976.0),
    K(105258076.0),
    K(45995730.0),
    K(13339535.0),
    K(2637558.0),
    K(357423.0),
    K(32670.0),
    K(1925.0),
    K(66.0),
    K(1.0)
  };*/
  static const R g = K(6.024680040776729583740234375000000000000);
#endif

static inline R evaluate_rational(const R z_)
{
  R z = z_, s1, s2;
  INT i;

  if (z <= K(1.0))
  {
    s1 = num[N - 1];
    s2 = K(1.0);
    for (i = N - 2; i >= 0; --i)
    {
      s1 *= z;
      s2 *= z + (R)(i);
      s1 += num[i];
    }
  }
  else
  {
    z = K(1.0)/z;
    s1 = num[0];
    s2 = K(1.0);
    for (i = 1; i < N; ++i)
    {
      s1 *= z;
      s2 *= K(1.0) + (R)(i-1) * z;
      s1 += num[i];
    }
  }
  return s1 / s2;
}

R Y(lambda)(const R z, const R eps)
{
  const R d = K(1.0) - eps, zpg = z + g, emh = eps - K(0.5);
  return EXP(-LOG1P(d / (zpg + emh)) * (z + emh)) *
    POW(KE / (zpg + K(0.5)),d) *
    (evaluate_rational(z + eps) / evaluate_rational(z + K(1.0)));
}

R Y(lambda2)(const R mu, const R nu)
{
  if (mu == K(0.0))
    return K(1.0);
  else if (nu == K(0.0))
    return K(1.0);
  else
    return
      SQRT(
        POW((mu + nu + g + K(0.5)) / (K(1.0) * (mu + g + K(0.5))), mu) *
        POW((mu + nu + g + K(0.5)) / (K(1.0) * (nu + g + K(0.5))), nu) *
        SQRT(KE * (mu + nu + g + K(0.5)) /
          ((mu + g + K(0.5)) * (nu + g + K(0.5)))) *
        (evaluate_rational(mu + nu + K(1.0)) /
          (evaluate_rational(mu + K(1.0)) * evaluate_rational(nu + K(1.0))))
      );
}
